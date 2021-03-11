# Place list of files to a csv
#     Source:  in D:\Visible_Data\3.SplitTrainVal\[Train|Val]\[1|2|3|4|m|ma] contain labelled, cropped images
#     Dest: 'ListLabelledFiles.csv'

$src_folders = "A:/RetellectImages/TrainValTest/Val","A:/RetellectImages/UnKnown"

$results_file = 'ListFiles.csv'

# empty out results file
$text | Set-Content $results_file



Foreach ($src_folder in $src_folders){
    get-childitem $src_folder -recurse |
        where-object { ($_.Extension -in '.jpg','.png')  } |
        #select-object FullName |
        ForEach-Object {

            # parse test/val and sub-category
            $set_name = $_.DirectoryName.Split('\')[-2]
            $barcode = $_.DirectoryName.Split('\')[-1]

            $_.FullName | Add-Content $results_file
        }
}