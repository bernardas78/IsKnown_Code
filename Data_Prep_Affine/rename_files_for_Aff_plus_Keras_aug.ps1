# copy by renaming files (to avoid name clash between Keras_Aug and AffAug file names)
#   src sample: A:\IsKnown_Images\Aff_NE_Balanced\v62\Ind-0\Train\18002\*.jpg
#   dst sample: A:\IsKnown_Images\Aff_NE_Balanced_AffineAndKerasAugCombined\affAug30_v62\Ind-0\Train\18002\*.jpg

$src_folder = 'A:\IsKnown_Images\Aff_NE_Balanced\v62'
#$dest_folder = 'A:\IsKnown_Images\Aff_NE_Balanced_AffineAndKerasAugCombined'

#$AffAugs = 10,20,30
$cntr = 0

#
Get-ChildItem $src_folder -Recurse -File -Filter "202*" |   #only rename unaugmented files which start with date; augmented filenames are unique
foreach {
    $name_no_ext = [io.path]::GetFileNameWithoutExtension( $_.Name)
    #$barcode = $_.FullName.Split('\')[-2]
    #$trainvaltest = $_.FullName.Split('\')[-3]
    #$hier = $_.FullName.Split('\')[-4]
    #$ver = $_.FullName.Split('\')[-5]
    $newfilename = $name_no_ext + '_KerAug.' + $_.Name.Split('.')[-1]    # add suffix "_A"

    Rename-Item $_.FullName $newfilename

    $cntr = $cntr+1
    if ($cntr%1000 -eq 0){
        echo ("Renamed " + $cntr + " files")
    }
    #foreach ($affAug in $AffAugs) {
    #    $affAug_ver_folder = [system.String]::Join("", ("affAug",$affAug,"_",$ver))
    #    $fullpath = [system.String]::Join("\",  ($dest_folder, $affAug_ver_folder, $hier, $trainvaltest, $barcode ))

    #    If (-not (Test-Path $fullpath)) {
    #        New-Item -ItemType Directory -Path $fullpath -Force
    #    }
    #    $newfile_fullpath = [system.String]::Join("\", ($fullpath,$newfilename))

    #    #Write-Host ($newfile_fullpath)
    #    Copy-Item $_.FullName -Destination $newfile_fullpath #Recurse creates dir if not exists
    #    #echo f | xcopy $_.FullName $newfile_fullpath
    #    $cntr = $cntr+1
    #    if ($cntr%1000 -eq 0){
    #        echo ("Copied " + $cntr + " files")
    #    }
    #}
}