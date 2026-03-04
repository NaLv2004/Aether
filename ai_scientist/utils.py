import logging
import colorlog
import os

def setup_logger(log_file_path):
    """
    初始化全局 Logger，同时输出到控制台(带颜色)和文件(纯文本)
    """
    # 确保日志文件夹存在
    # os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    logger = logging.getLogger("AgentLogger")
    logger.setLevel(logging.INFO)
    
    # 防止重复添加 Handler 导致日志打印多次
    if not logger.handlers:
        # 1. 控制台
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(colorlog.ColoredFormatter(
            '%(log_color)s[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S',
            log_colors={'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow', 'ERROR': 'red'}
        ))
        logger.addHandler(console_handler)

        # 2. 文件
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)

    return logger