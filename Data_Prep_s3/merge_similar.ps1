$dest_folder = 'A:\IsKnown_Images\Affine_NE_SameMrg\'

cd D:\IsKnown_Code\Data_Prep_s3

Import-Csv -Path 'merged_by_similarity.csv'  | 
#Import-Csv -Path 'merged_by_similarity.csv' -Header "newcode","barcode" | 
ForEach { 
    #echo $_.barcode
    #echo $_.newcode 

    $dest_folder_newcode = $dest_folder + $_.newcode
    $src_folder_barcode = $dest_folder + $_.barcode
    $src_pattern_barcode = $dest_folder + $_.barcode + '\*.*'
    #echo $dest_folder_newcode
    if (!(Test-Path -Path $dest_folder_newcode)){
        echo 'not exists'$dest_folder_newcode
        New-Item -Path $dest_folder -Name $_.newcode -ItemType "directory" 
        }

    Move-Item -Path $src_pattern_barcode -Destination $dest_folder_newcode

    Remove-Item -Path $src_folder_barcode
} 