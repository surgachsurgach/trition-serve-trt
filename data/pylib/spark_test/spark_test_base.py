import logging
import os
import sys
import warnings

from absl.testing import absltest
from pyspark import conf
from pyspark import sql


def _supress_py4j_logging():
    logger = logging.getLogger("py4j")
    logger.setLevel(logging.WARN)


# Disable ResourceWarning logs. For the detail. please refer to
# https://stackoverflow.com/questions/26563711/disabling-python-3-2-resourcewarning.
def _add_warning_filter():
    warnings.simplefilter("ignore")


def _create_pyspark_session(app_name: str) -> sql.SparkSession:
    spark_conf = conf.SparkConf()
    spark_conf.setMaster("local[*]")
    spark_conf.setAppName(app_name)
    spark_conf.set("spark.driver.host", "localhost")
    spark_conf.set("spark.pyspark.python", sys.executable)

    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    return sql.SparkSession.builder.config(conf=spark_conf).getOrCreate()


class SparkTestBase(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        _supress_py4j_logging()
        _add_warning_filter()

        cls.spark = _create_pyspark_session(cls.__name__)

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()
