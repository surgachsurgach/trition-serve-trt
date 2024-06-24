from absl import flags

flags.DEFINE_multi_string("gin_file", None, "List of paths to the config files.")
flags.DEFINE_multi_string("gin_param", None, "Newline separated list of Gin parameter bindings.")

FLAGS = flags.FLAGS
