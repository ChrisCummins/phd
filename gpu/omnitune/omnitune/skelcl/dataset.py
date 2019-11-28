from weka.filters import Filter as WekaFilter

from labm8.py import ml


class Dataset(ml.Dataset):

  def split_synthetic_real(self, db):
    """
    Split dataset based on whether scenario is a synthetic kernel or
    not.

    Returns:

       (WekaInstances, WekaInstances): Only instances from
         synthetic and real benchmarks, respectively.
    """
    real_scenarios = db.real_scenarios

    synthetic = self.copy(self.instances)
    real = self.copy(self.instances)

    # Loop over all instances from last to first.
    for i in range(self.instances.num_instances - 1, -1, -1):
      instance = self.instances.get_instance(i)
      scenario = instance.get_string_value(0)
      if scenario in real_scenarios:
        real.delete(i)
      else:
        synthetic.delete(i)

    return synthetic, real

  def arch_folds(self, db):
    """
    Split dataset to a list of leave-one-out instances, one for each
    architecture.

    Returns:

       list of (WekaInstances, WekaInstances) tuples: A list of
         training, testing pairs, where the training instances
         exclude all scenarios from a specific architecture, and
         the testing instances include only that architecture..
    """
    folds = []

    for device in db.devices:
      device_scenarios = db.scenarios_for_device(device)
      testing = self.copy(self.instances)
      training = self.copy(self.instances)

      # Loop over all instances from last to first.
      for i in range(self.instances.num_instances - 1, -1, -1):
        instance = self.instances.get_instance(i)
        scenario = instance.get_string_value(0)
        if scenario in device_scenarios:
          training.delete(i)
        else:
          testing.delete(i)

      folds.append((training, testing))

    return folds

  def kernel_folds(self, db):
    """
    Split dataset to a list of leave-one-out instances, one for each
    architecture.

    Returns:

       list of (WekaInstances, WekaInstances) tuples: A list of
         training, testing pairs, where the training instances
         exclude all scenarios from a specific kernel, and the
         testing instances include only that kernel.
    """
    folds = []

    for kernel in db.kernels:
      kernel_scenarios = db.scenarios_for_kernel(kernel)
      testing = self.copy(self.instances)
      training = self.copy(self.instances)

      # Loop over all instances from last to first.
      for i in range(self.instances.num_instances - 1, -1, -1):
        instance = self.instances.get_instance(i)
        scenario = instance.get_string_value(0)
        if scenario in kernel_scenarios:
          training.delete(i)
        else:
          testing.delete(i)

      folds.append((training, testing))

    return folds

  def dataset_folds(self, db):
    """
    Split dataset to a list of leave-one-out instances, one for each
    architecture.

    Returns:

       list of (WekaInstances, WekaInstances) tuples: A list of
         training, testing pairs, where the training instances
         exclude all scenarios from a specific dataset, and the
         testing instances include only that dataset.
    """
    folds = []

    for dataset in db.datasets:
      dataset_scenarios = db.scenarios_for_dataset(dataset)
      testing = self.copy(self.instances)
      training = self.copy(self.instances)

      # Loop over all instances from last to first.
      for i in range(self.instances.num_instances - 1, -1, -1):
        instance = self.instances.get_instance(i)
        scenario = instance.get_string_value(0)
        if scenario in dataset_scenarios:
          training.delete(i)
        else:
          testing.delete(i)

      folds.append((training, testing))

    return folds

  @staticmethod
  def load(path, db):
    nominals = [
        49,  # dev_double_fp_config
        50,  # dev_endian_little
        51,  # dev_execution_capabilities
        52,  # dev_extensions
        54,  # dev_global_mem_cache_type
        57,  # dev_host_unified_memory
        63,  # dev_image_support
        65,  # dev_local_mem_type
        96,  # dev_queue_properties
        97,  # dev_single_fp_config
        98,  # dev_type
        100,  # dev_vendor_id
    ]
    nominal_indices = ",".join([str(index) for index in nominals])
    force_nominal = ["-N", nominal_indices]

    # Load data from CSV.
    dataset = Dataset.load_csv(path, options=force_nominal)
    dataset.__class__ = Dataset

    # Set class index and database connection.
    dataset.class_index = -1
    dataset.db = db

    # Create string->nominal type attribute filter, ignoring the first
    # attribute (scenario ID), since we're not classifying with it.
    string_to_nominal = WekaFilter(classname=("weka.filters.unsupervised."
                                              "attribute.StringToNominal"),
                                   options=["-R", "2-last"])
    string_to_nominal.inputformat(dataset.instances)

    # Create filtered dataset, and swap data around.
    filtered = string_to_nominal.filter(dataset.instances)
    dataset.instances = filtered

    return dataset


class RegressionDataset(Dataset):

  @staticmethod
  def load(path, db):
    nominals = [
        49,  # dev_global_mem_cache_type
        52,  # dev_host_unified_memory
        54,  # dev_local_mem_type
        56,  # dev_type
        57,  # dev_vendor
    ]
    nominal_indices = ",".join([str(index) for index in nominals])
    force_nominal = ["-N", nominal_indices]

    # Load data from CSV.
    dataset = Dataset.load_csv(path, options=force_nominal)
    dataset.__class__ = Dataset

    # Set class index and database connection.
    dataset.class_index = -1
    dataset.db = db

    # Create string->nominal type attribute filter, ignoring the first
    # attribute (scenario ID), since we're not classifying with it.
    string_to_nominal = WekaFilter(classname=("weka.filters.unsupervised."
                                              "attribute.StringToNominal"),
                                   options=["-R", "2-last"])
    string_to_nominal.inputformat(dataset.instances)

    # Create filtered dataset, and swap data around.
    filtered = string_to_nominal.filter(dataset.instances)

    # Create nominal->binary type attribute filter, ignoring the
    # first attribute (scenario ID), since we're not classifying with it.
    n2b = WekaFilter(
        classname="weka.filters.unsupervised.attribute.NominalToBinary",
        options=["-R", "2-last"])
    n2b.inputformat(filtered)

    dataset.instances = n2b.filter(filtered)

    return dataset
