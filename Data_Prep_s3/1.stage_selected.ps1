#   Src: \<any hier str>\barcode\datetime-barcode-name.jpg ==>
#   Dest: selected barcodes only
#            \barcode\orig_filename.jpg


$source_pattern = 'A:\S3\photo\*.jpg'
$dest_folder = 'A:\IsKnown_Images\Selected_s3\'

cd D:\IsKnown_Code\Data_Prep_s3

$barcodes = Import-Csv -Path 'selected.csv' -Header "barcode" | ForEach { $_.barcode }

$i=0

# skip files before this date in S3 (instruction: cameras moved)
Get-ChildItem $source_pattern -Recurse -File | #Where {$_.FullName -notlike '*\2022-03-11\*'} |
Foreach{ 
    $barcode = $_.FullName.Split('\')[-2]

    if ( $_.FullName.Split('\')[-3] -like 'SCO*')
    {
        $store_pos = $_.FullName.Split('\')[-4] + "_" + $_.FullName.Split('\')[-3]
    }
    else
    {
        $store_pos = $_.FullName.Split('\')[-5] + "_" + $_.FullName.Split('\')[-4]
    }

    if ( $barcodes.Contains( $barcode ) ){

        $dest_folder_full = $dest_folder + $store_pos + "\" + $barcode

        if (! (Test-Path $dest_folder_full)) { 
            md  $dest_folder_full
        } 

        $full_newname = $dest_folder_full + '\' + $_.Name
        if (! (Test-Path -Path $full_newname -PathType Leaf)){
            copy $_.FullName $dest_folder_full
        }
    }
    $i++
    if ($i%1000 -eq 0)
    {
        echo $i
    }
}