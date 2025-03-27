
# Run one or all of the test cases in this directory

# Note: The appropriate Python environment must be activated prior to running this bash script

# Specify which test cases to run
run_basic='True'
run_localize='True'

# Option to keep test YAML files
keep_yaml='False'


#---------------------------------------------------------------------------------------------------

yml_dir=$( pwd )
cd ..

if [[ ${run_basic} == 'True' ]]; then
  echo
  echo "Running basic EnKF test case..."
  yml_name='./tests/basic_test.yml'
  test_dir='./tests/test_out'
  cp ./tests/S_NewEngland_2022020121_EnKF_test_input_TEMPLATE.yml ${yml_name}
  sed -i "s={TEST_DIR}=${yml_dir}=" ${yml_name}
  if [[ -d ${test_dir} ]]; then
    rm -r ${test_dir}
  fi
  mkdir ${test_dir}
  python ceilometer_obs_enkf.py ${yml_name}
  if [[ ${keep_yaml} == 'False' ]]; then
    rm ${yml_name}
  fi
fi

if [[ ${run_localize} == 'True' ]]; then
  echo
  echo "Running localization EnKF test case..."
  yml_name='./tests/localize_test.yml'
  test_dir='./tests/test_localize_out'
  cp ./tests/S_NewEngland_2022020121_EnKF_test_input_localization_TEMPLATE.yml ${yml_name}
  sed -i "s={TEST_DIR}=${yml_dir}=" ${yml_name}
  if [[ -d ${test_dir} ]]; then
    rm -r ${test_dir}
  fi
  mkdir ${test_dir}
  python ceilometer_obs_enkf.py ${yml_name}
  if [[ ${keep_yaml} == 'False' ]]; then
    rm ${yml_name}
  fi
fi
