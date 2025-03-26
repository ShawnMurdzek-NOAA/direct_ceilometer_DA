# Direct Ceilometer DA Tests

## Basic Functional Test (Also double as example cases)

To run, do the following:

1. Load Python environment
2. Edit the top section of `run_tests.sh` if desired
3. Run `bash run_tests.sh`

By default, `run_tests.sh` will run two test cases:

1. A "basic" test without localization. This test includes two experiments, one with 1 ceilometer observation and one with 3 ceilometer observations. The input YAML template is `S_NewEngland_2022020121_EnKF_test_input_TEMPLATE.yml`. Output should ideally match the output in `data/truth`.
2. Same as (1), but with localization included. The input YAML template is `S_NewEngland_2022020121_EnKF_test_input_localization_TEMPLATE.yml`. Output should ideally match the output in `data/localization_truth`.
