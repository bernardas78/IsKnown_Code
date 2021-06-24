import py7zr
import os

src_dir = r"A:\IsKnown_Images\Mrg_A_NE_BalKerasAff102030\Bal_v14"
dest_zip_dir = r"A:\IsKnown_Images\Mrg_A_NE_BalKerasAff102030_ZIP"

#curr_dir_to_exclude = "conf_mat_26"

for cur_dir in os.listdir(src_dir):
    src_dir_full = os.path.join ( src_dir, cur_dir )
    dest_zip = os.path.join ( dest_zip_dir, cur_dir+".7z" )

    if not os.path.exists (dest_zip): #and cur_dir!=curr_dir_to_exclude:
        print ("{} to {}".format(src_dir_full, dest_zip))

        #src_dir_full = r"A:\IsKnown_Images\Mrg_A_NE_BalKerasAff102030\Bal_v14\conf_mat_187"
        #dest_zip = r"A:\IsKnown_Images\Mrg_A_NE_BalKerasAff102030\Bal_v14\conf_mat_187.7z"

        with py7zr.SevenZipFile(dest_zip, 'w') as archive:
            archive.writeall(src_dir_full, cur_dir)
