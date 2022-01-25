"""
@author: Hongyi Zhou, uokad@student.kit.edu
@brief: WandbLogger, adapted from Ge Li's NMP repository
"""
import wandb
import os
import shutil
from datetime import datetime

def get_file_names_in_directory(directory: str):
    """
    Get file names in given directory
    Args:
        directory: directory where you want to explore

    Returns:
        file names in list

    """
    file_names = None
    try:
        (_, _, file_names) = next(os.walk(directory))
    except StopIteration as e:
        print("Cannot read files from directory: ", directory)
        exit()
    return file_names

def get_formatted_date_time():
    """
    Get formatted date and time, e.g. May-05-2021-22-14-31
    Returns:
        dt_string: date time string
    """
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
    return dt_string

def get_log_dir(log_name: str):
    """
    Get the dir to the log
    Args:
        log_name: log's name

    Returns:
        directory to log file
    """

    return os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        "log",
                        log_name)

def remove_file_dir(path):
    """
    Remove file or directory
    Args:
        path: path to directory or file

    Returns:
        True if successfully remove file or directory

    """
    if not os.path.exists(path):
        return False
    elif os.path.isfile(path) or os.path.islink(path):
        os.unlink(path)
        return True
    else:
        shutil.rmtree(path)
        return True

class WandbLogger:
    def __init__(self, config):
        self.project_name = config["logger_name"]
        self._initialize_log_dir()
        self._run = wandb.init(project=self.project_name,
                               config=config)
    
    def _initialize_log_dir(self):
        remove_file_dir(self.log_dataset_dir)
        remove_file_dir(self.log_model_dir)
        remove_file_dir(self.log_dir)

        os.makedirs(self.log_dir)
        os.makedirs(self.log_model_dir)
        os.makedirs(self.log_dataset_dir)
        
    @property
    def config(self):
        return wandb.config


    @property
    def log_dir(self):
        """
        Get local log saving directory
        Returns:
            log directory
        """
        return get_log_dir(self.project_name)


    @property
    def log_dataset_dir(self):
        """
        Get downloaded logged dataset directory
        Returns:
            logged dataset directory
        """
        return self.log_dir + "/dataset"


    @property
    def log_model_dir(self):
        """
        Get downloaded logged model directory
        Returns:
            logged model directory
        """
        return self.log_dir + "/model"

    def log_info(self, epoch, key, value):
        self._run.log({"Iteration": epoch,
                       key: value})

    def log_model(self,
                  finished: bool = False):
        """
        Log model into Artifact

        Args:
            finished: True if current training is finished, this will clean
            the old model version without any special aliass

        Returns:
            None
        """
        # Initialize wandb artifact
        model_artifact = wandb.Artifact(name="model", type="model")

        # Get all file names in log dir
        file_names = get_file_names_in_directory(self.log_model_dir)

        # Add files into artifact
        for file in file_names:
            path = os.path.join(self.log_model_dir, file)
            model_artifact.add_file(path)

        if finished:
            aliases = ["latest",
                       "finished-{}".format(get_formatted_date_time())]
        else:
            aliases = ["latest"]

        # Log and upload
        self._run.log_artifact(model_artifact, aliases=aliases)

        if finished:
            self.delete_useless_model()

    def delete_useless_model(self):
        """
        Delete useless models in WandB server
        Returns:
            None

        """
        api = wandb.Api()

        artifact_type = "model"
        artifact_name = "{}/{}/model".format(self._run.entity,
                                             self._run.project)

        for version in api.artifact_versions(artifact_type, artifact_name):
            # Clean up all versions that don't have an alias such as 'latest'.
            if len(version.aliases) == 0:
                version.delete()


    def load_model(self,
                   model_api: str):
        """
        Load model from Artifact

        model_api: the string for load the model if init_epoch is not zero

        Returns:
            model_dir: Model's directory

        """
        model_api = "self._" + model_api[11:]
        artifact = eval(model_api)
        artifact.download(root=self.log_model_dir)
        file_names = get_file_names_in_directory(self.log_model_dir)
        file_names.sort()
        for file in file_names:
            print(file)
        return self.log_model_dir

if __name__ == "__main__":
    config = {}
    sub_dict = {}
    sub_dict["log_name"] = "test_exp"
    config["logger"]= sub_dict
    logger = WandbLogger(config)
    print(logger.log_dir)