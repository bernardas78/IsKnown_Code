# Makes 5 datasets 
#
#  Source: TrainValTest\[Train|Val|Test]
#
#  Dest: \Ind-[0:4]\[Train|Val|Test], where 0:4 indicates number of characters subtracted from the right of Product code

$Ind_minus = 0

#$src_folder = 'A:\IsKnown_Images\TrainValTest'
$src_folder = 'A:\IsKnown_Images\A_M_TrainValTest'

#$dest_folder_pattern = 'A:\IsKnown_Images\Hier\Ind-'
$dest_folder_pattern = 'A:\IsKnown_Images\A_M_Hier\Ind-'


for ($Ind_minus = 0; $Ind_minus -lt 5; $Ind_minus++)
{
    $dest_folder = $dest_folder_pattern+$Ind_minus

    Get-ChildItem $src_folder -File -Recurse |
    Foreach{ 
        $barcode = $_.FullName.Split("\")[-2]
        $trainValTest = $_.FullName.Split("\")[-3]
        #echo $barcode, $trainValTest

        # class_code - hierarchical category code (e.g. Ind.prod: 18035, then class_code: 1803, 180, 18, 1
        $class_code = $barcode.Substring(0, $barcode.Length-$Ind_minus)

        $dest_folder_full = $dest_folder + "\" + $trainValTest + "\" + $class_code
        #echo $dest_folder_full

        if (! (Test-Path $dest_folder_full)) { 
            md  $dest_folder_full
        } 

        xcopy $_.FullName $dest_folder_full /q /y
    }
}