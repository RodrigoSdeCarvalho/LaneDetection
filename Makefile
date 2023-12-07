default: main

train:
	cd app && python3 train_enet.py

test:
	cd app && python3 center_of_lane_test.py

video:
	cd app && python3 video_test.py

fps:
	cd app && python3 fps_test.py

freeze:
	pipreqs ./

clear-logs:
	rm -rf logs/*
	touch logs/.gitkeep
	clear
	echo "Logs cleared"
