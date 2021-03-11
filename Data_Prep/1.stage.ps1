# Splits by barcode
#   Filters by dates

#   Src: \datetime-barcode-name.jpg ==> 
#   Dest: \barcode\orig_filename.jpg

$source_pattern = 'A:\AK Dropbox\n20190113 A\Justiniskes\2?\camera1\*.jpg'

#$from_date = '20000101' # >=
#$to_date = '20201210'   # <
$from_date = '20201210' # >=
$to_date = '20210203'   # <

#$dest_folder = 'A:\RetelectImages\Staging\'
$dest_folder = 'A:\RetellectImages\Staging_AllUpTo20210203\'

Get-ChildItem $source_pattern |
    Where-Object Name -gt $from_date% |
    Where-Object Name -lt $to_date% |  
Foreach{ 
    $barcode = $_.Name.Split('-')[1]
    $product_name = $_.Name.Split('-')[2].split('.')[0]
    $sco = $_.FullName.Split('\')[-3]
    #echo $_.Name

    $dest_folder_full = $dest_folder + $sco + "\" + $barcode
    #echo $dest

    if (! (Test-Path $dest_folder_full)) { 
        md  $dest_folder_full
    } 

    copy $_.FullName $dest_folder_full

}