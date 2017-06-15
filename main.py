from learningpgn import LearningPGN
from learning import Learning
import logging

import datetime
import time

path_log = "logs"
log_file = str(datetime.datetime.now())
logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logging.basicConfig(
    filename="{0}/chess-ai-{1}.log".format(path_log, log_file), level=logging.DEBUG)
logging.getLogger().addHandler(consoleHandler)
logger = logging.getLogger(__name__)


def main():
    start_time = time.time()
    learning = LearningPGN()
    # learning = Learning()
    learning.learn()
    logger.info("Time process %s" % (time.time() - start_time))


if __name__ == "__main__":
    main()
