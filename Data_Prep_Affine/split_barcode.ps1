# Creates folder-structure for images:
#   Src: \datetime-barcode-name.jpg ==>
#   Dest: \barcode\orig_filename.jpg

$source_folder = 'A:\IsKnown_Images\Selected_TimeSplitTmp'
$dest_folder = 'A:\IsKnown_Images\Selected_TimeBarcodeSplit'

Get-ChildItem $source_folder -File -Recurse |
Foreach{
    $sco_time = $_.FullName.Split('\')[-2]
    $barcode = $_.Name.Split('-')[1]
    #echo $sco_time

    # substring (3,5)
    $barcode = $barcode.Substring(2,5)

    $dest_folder_full = $dest_folder + '\' + $sco_time + '\' + $barcode
    #echo $dest_folder_full
    if (! (Test-Path $dest_folder_full)) {
        echo $dest_folder_full
        md  $dest_folder_full
    }

    copy $_.FullName $dest_folder_full

}