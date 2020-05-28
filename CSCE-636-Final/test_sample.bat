@echo off
set VIDEO=Final/v_Steeple_503.avi
set UINJSON=124006988.json
set UINJPG=124006988.jpg
set JSON=data.json
set JPG=data.jpg
echo %VIDEO%
py -3.6 video_test.py --video_name %VIDEO%

rem rename the generated timeLabel.json and figure with your UIN.
copy %JPG% %UINJPG%
copy %JSON% %UINJSON%
