
# Create Ceilometer DA Inputs

tmpl_yml="S_NewEngland_2022020121_EnKF_input.yml"
final_yml=${tmpl_yml}
code_dir="/ncrc/home2/Shawn.S.Murdzek/src/direct_ceilometer_DA"
run_script="${code_dir}/drivers/run_ceilometer_DA.sh"
out_dir=`pwd`

# State variables, delimited with a semicolon
state=( 'TMP_P0_L105_GLC0;SPFH_P0_L105_GLC0'
	'TMP_P0_L105_GLC0;SPFH_P0_L105_GLC0;CLWMR_P0_L105_GLC0;ICMR_P0_L105_GLC0' 
        'TMP_P0_L105_GLC0;SPFH_P0_L105_GLC0;UGRD_P0_L105_GLC0;VGRD_P0_L105_GLC0' )

# Localization
# lh_all and lv_all must have the same length
# Use -1 for no localization
lh_all=( -1 50 50 100 25 )
lv_all=( -1 4  1  4   4 )

# Option to redo H(x)
redo_hofx=( 'True' 'False' )

# Observation error variance
obvar=( 156.25 )

#-------------------------------------------------------------------------------

for s in ${state[@]}; do
  for rhofx in ${redo_hofx[@]}; do
    for il in ${!lh_all[@]}; do
      for o in ${obvar[@]}; do

        tmp_fname='tmp.yml'
        cp ${tmpl_yml} ${tmp_fname}

        pattern="  - 'TCDC_P0_L105_GLC0'"
        tag_state=""
        IFS=';' read -ra vars <<< "${s}"
        for v in ${vars[@]}; do
          sed -i "/${pattern}/a \  - '${v}'" ${tmp_fname}
          pattern="  - '${v}'"
          case ${v} in
            'TMP_P0_L105_GLC0')
              tag_state+='T'
            ;;
            'SPFH_P0_L105_GLC0')
              tag_state+='Qv'
            ;;
            'UGRD_P0_L105_GLC0')
              tag_state+='U'
            ;;
            'VGRD_P0_L105_GLC0')
              tag_state+='V'
            ;;
            'CLWMR_P0_L105_GLC0')
              tag_state+='Qc'
            ;;
            'ICMR_P0_L105_GLC0')
              tag_state+='Qi'
            ;;
          esac
        done

        sed -i "s={REDO_HOFX}=${rhofx}=" ${tmp_fname}
        if [[ ${rhofx} == 'True' ]]; then
          tag_hofx='T'
        else
          tag_hofx='F'
        fi

        if [[ ${lh_all[il]} -eq -1 ]]; then
          sed -i "s={LOCALIZATION}=False=" ${tmp_fname}
          tag_lh='N'
          tag_lv='N'
        else
          sed -i "s={LOCALIZATION}=True=" ${tmp_fname}
          sed -i "s={LH}=${lh_all[il]}=" ${tmp_fname}
          sed -i "s={LV}=${lv_all[il]}=" ${tmp_fname}
          tag_lh="${lh_all[il]}"
          tag_lv="${lv_all[il]}"
        fi

        sed -i "s={OBVAR}=${o}=" ${tmp_fname}

        out_dir="state_${tag_state}_lv${tag_lv}_lh${tag_lh}_obvar${o}_RedoHofX_${tag_hofx}"
        sed -i "s={OUTDIR}=${out_dir}=" ${tmp_fname}
        mkdir ${out_dir}
        mv ${tmp_fname} ${out_dir}/${final_yml}
        cp ${run_script} ${out_dir}
        cd ${out_dir}
        echo "Submitting job for ${out_dir}"
        sbatch ${run_script}
        cd ..

      done
    done
  done
done

