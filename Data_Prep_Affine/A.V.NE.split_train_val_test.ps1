# Source directory where barcode folders are located
$images_folder = 'A:\IsKnown_Images\Cleaned_Aff_NE_AutoVisible\'

# Destination directory
$sets_folder = 'A:\IsKnown_Images\Aff_NE_TrainValTest\'

$pct_train=0.64
$pct_val=0.16
$pct_test=0.2

# Expected structure: Cleaned_AutoVisible\v<version>\barcode\*.jpg

# For each version of Visible model
Get-ChildItem $images_folder -Directory |
    ForEach-Object { 
        
        $modelVersion_folder = $_.FullName

        $train_folder = $sets_folder + $_.Name + '\Train'
        $val_folder = $sets_folder + $_.Name +'\Val'
        $test_folder = $sets_folder + $_.Name +'\Test'


        # Recreate child folders to avoid errors
        Remove-Item $train_folder -Force -Recurse
        Remove-Item $val_folder -Force -Recurse
        Remove-Item $test_folder -Force -Recurse
        md $train_folder
        md $val_folder
        md $test_folder


        # Copy file to selected Train or Val or Test (64/16/20)
        Get-ChildItem $modelVersion_folder -Directory |
            ForEach-Object { 

                $barcode_folder = $_.FullName

                $file_cnt = (Get-ChildItem $barcode_folder | Measure-Object).Count
                # Skip <3 dirs
                If ($file_cnt -ge 3){
            
                    #temp counts how many files already copied to train, val, test sets
                    $actual_train=0
                    $actual_val=0
                    $actual_test=0

                    # copy each file to a proper set's folder
                    Get-ChildItem $barcode_folder |
                        ForEach-Object { 

                            #At least one image goes to each set: Train, Val, Test
                            $tot_copied = $actual_train + $actual_val + $actual_test + 1e-7
                            If ($pct_train - $actual_train/$tot_copied -gt $pct_val - $actual_val/$tot_copied -and
                                $pct_train - $actual_train/$tot_copied -gt $pct_test - $actual_test/$tot_copied)
                                {
                                    $actual_train+=1
                                    #echo "copy to train"
                                    $dest_folder = $train_folder
                                }
                            ElseIf ($pct_val - $actual_val/$tot_copied -gt $pct_test - $actual_test/$tot_copied )
                                {
                                    $actual_val+=1
                                    #echo "copy to val"
                                    $dest_folder = $val_folder
                                }
                            Else
                                {
                                    $actual_test+=1
                                    #echo "copy to test"
                                    $dest_folder = $test_folder
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
        }