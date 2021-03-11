# Crop files; keep structure \SCO\Barcode


$src_folder = "A:\IsKnown_Images\Selected\Ind-0\"

$dest_folder = "A:\IsKnown_Images\Cropped\"

Remove-Item $dest_folder -Force -Recurse
md $dest_folder

$crop_width_this =  480
$crop_height_this = 480
$crop_x_this = 0
$crop_y_this = 160

get-childitem $src_folder -recurse | 
    where-object { $_.Extension -in '.jpg','.png' } |
    select-object FullName,Name |
    ForEach-Object {
        
        $barcode = $_.FullName.Split('\')[-2]
        $sco = $_.FullName.Split('\')[-3]
        $hier = $_.FullName.Split('\')[-4]

        # create if not exists
        #$dest_folder_full = $dest_folder + $hier + "\" + $sco + "\" + $barcode
        $dest_folder_full = $dest_folder + $barcode
        If(!(test-path $dest_folder_full))
        {
              New-Item -ItemType Directory -Force -Path $dest_folder_full
        }

        $filename = $_.Name

        $dest = $dest_folder_full + "\" + $_.Name
        #echo $dest

        #Crop and save
        $cmd = "magick.exe convert -crop " +
                $crop_width_this + "x" + $crop_height_this + "+" + $crop_x_this + "+" + $crop_y_this + 
                " """ + $_.FullName + """ """ + $dest + """"

        #echo $cmd
        iex $cmd

        #magick.exe convert -crop 300x300+283+92 D:/Google Drive/PhD_Data/Raw/SCO1/1/000000005311_4_20191001152033817.jpg D:/Visible_Data/000000005311_4_20191001152033817.jpg
    }