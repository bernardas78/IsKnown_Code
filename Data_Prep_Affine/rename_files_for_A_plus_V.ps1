# copy by renaming files (to avoid name clash between A_Hier and V_Hier)

$src_folder = 'A:\IsKnown_Images\A_Hier'
$dest_folder = 'A:\IsKnown_Images\A_V_Hier'

Get-ChildItem $src_folder -Recurse -File |
foreach {
    $name_no_ext = [io.path]::GetFileNameWithoutExtension( $_.Name)
    $newfilename = $name_no_ext + '_A.' + $_.Name.Split('.')[-1]    # add suffix "_A"
    $ver_hier_barcode = [system.String]::Join("\", $_.DirectoryName.Split('\')[-4..-1] ) # version\hier-lvl\barcode\
    $newfile_fullpath = [system.String]::Join("\",  ($dest_folder, $ver_hier_barcode, $newfilename ))
    #echo $newfile_fullpath
    #Copy-Item $_.FullName -Destination $newfile_fullpath
    echo f | xcopy $_.FullName $newfile_fullpath
}