
# Run one or all of the test cases in this directory

# Note: The appropriate Python environment must be activated prior to running this bash script

# Specify which test cases to run
run_basic=True
run_localize=True

# Option to keep test YAML files
keep_yaml=False


#---------------------------------------------------------------------------------------------------

# Add direct_ceilometer_DA to PYTHONPATH
cd ../../
export PYTHONPATH=$PYTHONPATH:$( pwd )
cd direct_ceilometer_DA/tests

home=$( pwd )

if [[ ${run_basic} ]]; then
  echo
  echo "Running basic EnKF test case..."
  yml_name='basic_test.yml'
  test_dir='test_out'
  cp S_NewEngland_2022020121_EnKF_test_input_TEMPLATE.yml ${yml_name}
  sed -i "s={TEST_DIR}=${home}=" ${yml_name}
  if [[ -d ${test_dir} ]]; then
    rm -r ${test_dir}
  fi
  mkdir ${test_dir}
  python ../drivers/ceilometer_obs_enkf.py ${yml_name}
  if [[ ! ${keep_yaml} ]]; then
    rm ${yml_name}
  fi
fi

if [[ ${run_localize} ]]; then
  echo
  echo "Running localization EnKF test case..."
  yml_name='localize_test.yml'
  test_dir='test_localize_out'
  cp S_NewEngland_2022020121_EnKF_test_input_localization_TEMPLATE.yml ${yml_name}
  sed -i "s={TEST_DIR}=${home}=" ${yml_name}
  if [[ -d ${test_dir} ]]; then
    rm -r ${test_dir}
  fi
  mkdir ${test_dir}
  python ../drivers/ceilometer_obs_enkf.py ${yml_name}
  if [[ ! ${keep_yaml} ]]; then
    rm ${yml_name}
  fi
fi
