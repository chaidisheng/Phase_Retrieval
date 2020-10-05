import os
import time
import logging
import subprocess


def init_logging(log_path):
    r"""TODO: Docstring for init_logger.
    Args:
        :log_path: path of store
        :return: logger
    Examples:
        >>>
    """
    # first step: 第一步，创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # second step: 第二步，创建一个handler，用于写入日志文件
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    logfile = log_path + rq + '_info.log'
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)
    # third step: 第三步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # fourth step: 第四步，将logger添加到handler里面
    logger.addHandler(fh)

    return logger


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def get_git_revision_short_hash():
    r"""Returns the current Git commit.
    """
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()


def init_logger(argdict):
    r"""Initializes a logging.Logger to save all the running parameters to a
    log file

    Args:
        argdict: dictionary of parameters to be logged
    """

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    fh = logging.FileHandler(os.path.join(argdict.log_dir, 'log.txt'), mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    try:
        logger.info("Commit: {}".format(get_git_revision_short_hash()))
    except Exception as e:
        logger.error("Couldn't get commit number: {}".format(e))
    logger.info("Arguments: ")
    for k in argdict.__dict__:
        logger.info("\t{}: {}".format(k, argdict.__dict__[k]))

    return logger


def init_logger_ipol():
    r"""Initializes a logging.Logger in order to log the results after
    testing a model

    Args:
        result_dir: path to the folder with the denoising results
    """
    logger = logging.getLogger('testlog')
    logger.setLevel(level=logging.INFO)
    fh = logging.FileHandler('out.txt', mode='w')
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def init_logger_test(result_dir):
    r"""Initializes a logging.Logger in order to log the results after testing
    a model

    Args:
        result_dir: path to the folder with the denoising results
    """
    logger = logging.getLogger('testlog')
    logger.setLevel(level=logging.INFO)
    fh = logging.FileHandler(os.path.join(result_dir, 'log.txt'), mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


if __name__ == '__main__':
    set_logger('/home/chaidisheng/Pycharm/PycharmProjects/denoiser_chai/utilities/log.py')