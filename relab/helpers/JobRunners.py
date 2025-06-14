import abc
import logging
import os
import re
import subprocess
import time
from concurrent.futures import Future, ProcessPoolExecutor, wait
from functools import partial
from typing import Callable, List

from relab.helpers.Typing import Config
from relab.scripts.draw_graph import draw_graph
from relab.scripts.run_demo import run_demo
from relab.scripts.run_training import run_training


class JobRunnerInterface(abc.ABC):
    """!
    Class that all job runners must implement.
    """

    @abc.abstractmethod
    def launch_job(
        self, task: str, kwargs: Config, dependencies: List[int] = None
    ) -> int:
        """!
        Launch a job.
        @param task: the task to run, e.g., "run_training" or "run_demo"
        @param kwargs: the keywords argument of the job's shell script
        @param dependencies: the job indices on which this job depends
        @return the index of the slurm job launched
        """
        ...

    def wait(self) -> None:
        """!
        Wait for all running and scheduled jobs to terminate, if applicable.
        """
        ...


class SlurmJobRunner(JobRunnerInterface):
    """!
    Class launching slurm jobs.
    """

    def launch_job(
        self, task: str, kwargs: Config, dependencies: List[int] = None
    ) -> int:
        """!
        Launch a slurm job.
        @param task: the task to run, e.g., "run_training" or "run_demo"
        @param kwargs: the keywords argument of the job's shell script
        @param dependencies: the job indices on which this job depends
        @return the index of the slurm job launched
        """

        # Create the job's arguments.
        args = []
        for key, value in kwargs.items():
            if isinstance(value, list):
                value = " ".join([str(v) for v in value])
            args.append(f"--{key} {value}")
        args = " ".join(args)

        # Create the job's dependencies.
        dependencies = [str(dependency) for dependency in dependencies]
        if dependencies is None:
            dependencies = ""
        else:
            dependencies = f"-d afterok:{':'.join(dependencies)}"

        # Create the job's command line.
        command = f"sbatch {dependencies} {task} {args}"

        # Launch the slurm job.
        process = subprocess.run(command.split(), capture_output=True, text=True)

        # Return the job index of the slurm job.
        return re.findall(r"\d+", process.stdout)[-1]


class LocalJobRunner(JobRunnerInterface):
    """!
    Class launching local jobs.
    """

    def __init__(self, max_worker: int = 1) -> None:
        """!
        Create a local job runner.
        @param max_worker: the maximum number of worker
        """

        # @var pool
        # The pool of workers for executing tasks concurrently.
        self.pool = ProcessPoolExecutor(max_workers=max_worker)

        # @var max_worker
        # The maximum number of concurrent workers allowed in the process pool
        self.max_worker = max_worker

        # @var futures
        # List of future objects returned when submitting tasks.
        self.futures = []

        # @var job_index
        # Counter for assigning unique job IDs, incremented with each new job.
        self.job_index = -1

        # @var lock_index
        # Counter used to synchronize access to shared resources.
        self.lock_index = -1

        # @var jobs_not_done
        # List of jobs that are currently in progress or pending execution.
        self.jobs_not_done = []

        # @var jobs_to_submit
        # Dictionary of jobs waiting for their dependencies to complete.
        self.jobs_to_submit = {}

    def set_lock_index(self, job_index: int = None) -> bool:
        """!
        Set the lock index, if the object is not currently locked.
        @param job_index: the new value of the lock index
        @return True if the lock index has been set, False if the object is currently locked
        """
        if self.lock_index == -1:
            self.lock_index = job_index
            return True
        return False

    def lock(self, job_index: int = None) -> None:
        """!
        Lock the object for a specific job index.
        @param job_index: the job index requesting the lock
        """

        # Wait until the object is available.
        while self.set_lock_index(job_index) is False:
            time.sleep(0.1)

    def unlock(self) -> None:
        """!
        Unlock the object.
        """
        self.lock_index = -1

    def satisfied(self, dependencies: List[int]) -> bool:
        """!
        Check whether all the dependencies have finished their execution.
        @param dependencies: the job indices whose execution needs to be checked
        @return True if all the dependencies have finished their execution, False otherwise
        """

        # Check if there are any dependencies.
        if dependencies is None:
            return True

        # Check if the dependencies are satisfied.
        for dependency in dependencies:
            if dependency in self.jobs_not_done:
                return False
        return True

    def submit(self, task: Callable, kwargs: Config, job_index: int) -> None:
        """!
        Submit a job to the pool (for internal use only).
        @param task: the task to run
        @param kwargs: the keyword arguments of the task
        @param job_index: the index of the job to run
        """
        # @cond IGNORED_BY_DOXYGEN
        logging.info(
            f"Submitting job[{job_index}], it will start when worker becomes available: {task}, {kwargs}"
        )
        future = self.pool.submit(task, **kwargs)
        future.add_done_callback(
            partial(self.check_jobs_to_submit, job_index=job_index)
        )
        self.futures.append(future)
        # @endcond

    def check_jobs_to_submit(self, future: Future, job_index: int) -> None:
        """!
        Check if new jobs can be submitted.
        @param future: the future corresponding to the job that just finished (unused)
        @param job_index: the index of the job that just finished
        """

        # Lock the object to avoid simultaneous access to the class attributes.
        logging.info(f"Job {job_index} just finished.")
        self.lock(job_index)

        # Remove the index of completed job from the list of jobs not done.
        self.jobs_not_done.remove(job_index)

        # Submit jobs whose dependencies are now satisfied.
        jobs_to_submit = {}
        for index, (task, kwargs, dependencies) in self.jobs_to_submit.items():
            if self.satisfied(dependencies):
                jobs_to_submit[index] = (task, kwargs, dependencies)
        for index, (task, kwargs, dependencies) in jobs_to_submit.items():
            del self.jobs_to_submit[index]
            self.submit(task, kwargs, index)

        # Unlock the object to avoid deadlock.
        self.unlock()

    def launch_job(
        self, task: str, kwargs: Config, dependencies: List[int] = None
    ) -> int:
        """!
        Launch a local job.
        @param task: the task to run, e.g., "run_training" or "run_demo"
        @param kwargs: the keywords argument of the job's shell script
        @param dependencies: the job indices on which this job depends
        @return the index of the slurm job launched
        """

        # Lock the object to avoid simultaneous access to the class attributes.
        self.lock()

        # Increase the job index, and add it to the list of jobs not done.
        self.job_index += 1
        self.jobs_not_done.append(self.job_index)

        # Retrieve the function to run.
        tasks = {
            "run_training": run_training,
            "run_demo": run_demo,
            "draw_graph": draw_graph,
        }
        task_name = re.findall(rf"([^{os.sep}]*).sh", task)[-1]
        task = tasks[task_name]

        # Check whether to submit the job directly.
        if not self.satisfied(dependencies):
            self.jobs_to_submit[self.job_index] = (task, kwargs, dependencies)
            self.unlock()
            return self.job_index

        # Launch the local job.
        self.submit(task, kwargs, self.job_index)

        # Return the job index.
        self.unlock()
        return self.job_index

    def wait(self) -> None:
        """!
        Wait for all running and scheduled jobs to terminate.
        """

        # Wait for all jobs to be submitted to the pool.
        while len(self.jobs_to_submit) != 0:
            time.sleep(0.1)

        # Wait for all running jobs to terminate.
        wait(self.futures)
        self.futures.clear()
