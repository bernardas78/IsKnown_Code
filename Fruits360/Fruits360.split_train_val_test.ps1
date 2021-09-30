# Test: entire              \fruits-360_dataset\fruits-360\Test folder (copy manually)
# Train+Val: 80/20 split    \fruits-360_dataset\fruits-360\Training

# Source directory where barcode folders are located
$images_folder = 'A:\Fruits360\fruits-360_dataset\fruits-360\Training\'

# Destination train directory
#$sets_folder = 'A:\IsKnown_Images\TrainValTest\'
$sets_folder = 'A:\Fruits360\TrainValTest\'
$train_folder = $sets_folder+'Train'
$val_folder = $sets_folder+'Val'


# Recreate child folders to avoid errors
Remove-Item $train_folder -Force -Recurse
Remove-Item $val_folder -Force -Recurse
md $train_folder
md $val_folder

$pct_train=0.8
$pct_val=0.2


# Copy file to randomly selected Train or Val (80/20)
Get-ChildItem $images_folder -Directory |
    ForEach-Object { 
        $file_cnt = (Get-ChildItem $_.FullName | Measure-Object).Count
        # Skip <3 dirs
        If ($file_cnt -ge 3){
            
            #temp counts how many files already copied to train, val sets
            $actual_train=0
            $actual_val=0

            # copy each file to a proper set's folder
            Get-ChildItem $_.FullName |
                ForEach-Object { 

                    #At least one image goes to each set: Train, Val
                    $tot_copied = $actual_train + $actual_val + 1e-7
                    If ($pct_train - $actual_train/$tot_copied -gt $pct_val - $actual_val/$tot_copied)
                        {
                            $actual_train+=1
                            #echo "copy to train"
                            $dest_folder = $train_folder
                        }
                    Else
                        {
                            $actual_val+=1
                            #echo "copy to val"
                            $dest_folder = $val_folder
                        }

                    # 80/20 split

                    #    [-1] is barcode (last directory)
                    $dest_folder += "\" + $_.DirectoryName.split('\')[-1] + "\"
                    #echo $_.DirectoryName,$_.DirectoryName.split('\')[-1],$dest_folder
    
                    #echo $_.FullName, $dest_folder
                    xcopy $_.FullName $dest_folder
                    #echo $_.Name
                }
            }
    }