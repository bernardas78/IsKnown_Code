# Makes 5 datasets (first import temp\prekes.csv into SQL dbo.imgcnt
#
#  Source: 2.select_prods.sql
#
#  Dest: \Ind-[0:4], where 0:4 indicates number of characters subtracted from the right of Product code

#$Ind_minus = 0

$Server = ".\sqlexpress"
$DB = "SCO"

#$src_folder = 'A:\RetellectImages\Staging_AllUpTo20201210'
$src_folder = 'A:\RetellectImages\Staging_AllUpTo20210203'

#$dest_folder_pattern = 'A:\IsKnown_Images\Selected\Ind-'
$dest_folder = 'A:\IsKnown_Images\Selected'

$MyDir = [System.IO.Path]::GetDirectoryName($myInvocation.MyCommand.Definition)
$input_File= $MyDir+"\2.select_prods.sql"
$Selected_Prods = Invoke-Sqlcmd -ServerInstance $Server -Database $DB -InputFile $input_File

#for ($Ind_minus = 0; $Ind_minus -lt 5; $Ind_minus++)
#{
#$dest_folder = $dest_folder_pattern+$Ind_minus

Get-ChildItem $src_folder -File -Recurse |
Foreach{
    #echo $_.FullName
    $barcode = $_.FullName.Split("\")[-2]
    #echo $barcode

    If ($Selected_Prods.FullBarcode.Contains($barcode)) {
        $sco = $_.FullName.Split("\")[-3]
        #echo $sco

        $class_code = ($Selected_Prods | Where FullBarcode -eq $barcode).DestFolder

        # subtract 0-4 characters from the right
        $shortened_class_code = $class_code.Substring(0, $class_code.Length-$Ind_minus)

        $dest_folder_full = $dest_folder + "\" + $sco + "\" + $shortened_class_code
        #echo $dest_folder_full

        if (! (Test-Path $dest_folder_full)) {
            md  $dest_folder_full
        }

        xcopy $_.FullName $dest_folder_full /q /y
    }
}
#}