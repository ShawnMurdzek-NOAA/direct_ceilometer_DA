
module load wgrib2

in_fname=rrfs.t03z.natlev.f001.conus.grib2
out_fname=rrfs.t03z.natlev.TEST.f001.conus.grib2

for i in $(seq 1 3); do
  cd mem00${i}
  # Extract TMP (4 hybrid levels), FRACCC (4 hybrid levels), SPFH (4 hybrid levels), HGT (4 hybrid levels), TKE (4 hybrid levels), and HGT of sfc
  wgrib2 ${in_fname} -match '^(14|34|54|74|12|32|52|72|15|35|55|75|13|33|53|73|19|39|59|79|1255|):' -grib ${out_fname}
  cd ..  
done
