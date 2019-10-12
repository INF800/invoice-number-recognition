# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2
from fuzzywuzzy import process, fuzz
import xlsxwriter
import glob
import os

# add path to tesseract here
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def decode_predictions(scores, geometry):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < args["min_confidence"]:
				continue

			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
	help="nearest multiple of 32 for resized width")
ap.add_argument("-e", "--height", type=int, default=320,
	help="nearest multiple of 32 for resized height")
ap.add_argument("-p", "--padding", type=float, default=0.0,
	help="amount of padding to add to each border of ROI")
args = vars(ap.parse_args())





# main
images_folder = glob.glob(os.getcwd() +'/images/*')

# this collection will be later used for generating excel file
# we will append details to it.
content = [['IMAGE NAME', 'INVOICE NUMBER']]

# looping over all images in 'images' folder
for f in images_folder:
	
	# for storing detected information of current image
	detection_details = []

	# get images names from folders 
	im_name = f.split('\\')[-1].split(".")[0]
	detection_details.append(im_name)
	print("[INFO] Working on {}.png".format(im_name))

	# load the input image and grab the image dimensions
	image = cv2.imread(f)
	orig = image.copy()
	(origH, origW) = image.shape[:2]

	# set the new width and height and then determine the ratio in change
	# for both the width and height
	(newW, newH) = (args["width"], args["height"])
	rW = origW / float(newW)
	rH = origH / float(newH)

	# resize the image and grab the new image dimensions
	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]

	# define the two output layer names for the EAST detector model that
	# we are interested -- the first is the output probabilities and the
	# second can be used to derive the bounding box coordinates of text
	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]

	# load the pre-trained EAST text detector
	print("[INFO] Loading EAST text detector...")
	print("[INFO] this may take some time...")
	net = cv2.dnn.readNet("EAST_model/frozen_east_text_detection.pb")

	# construct a blob from the image and then perform a forward pass of
	# the model to obtain the two output layer sets
	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)

	# decode the predictions, then  apply non-maxima suppression to
	# suppress weak, overlapping bounding boxes
	(rects, confidences) = decode_predictions(scores, geometry)
	boxes = non_max_suppression(np.array(rects), probs=confidences)

	# initialize the list of results
	results = []

	# loop over the bounding boxes
	for (startX, startY, endX, endY) in boxes:
		# scale the bounding box coordinates based on the respective
		# ratios
		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)

		# in order to obtain a better OCR of the text we can potentially
		# apply a bit of padding surrounding the bounding box -- here we
		# are computing the deltas in both the x and y directions
		dX = int((endX - startX) * args["padding"])
		dY = int((endY - startY) * args["padding"])

		# apply padding to each side of the bounding box, respectively
		startX = max(0, startX - dX)
		startY = max(0, startY - dY)
		endX = min(origW, endX + (dX * 2))
		endY = min(origH, endY + (dY * 2))

		# extract the actual padded ROI
		roi = orig[startY:endY, startX:endX]

		# in order to apply Tesseract v4 to OCR text we must supply
		# (1) a language, (2) an OEM flag of 4, indicating that the we
		# wish to use the LSTM neural net model for OCR, and finally
		# (3) an OEM value, in this case, 7 which implies that we are
		# treating the ROI as a single line of text
		config = ("-l eng --oem 1 --psm 7")
		text = pytesseract.image_to_string(roi, config=config)

		# add the bounding box coordinates and OCR'd text to the list
		# of results
		results.append(((startX, startY, endX, endY), text))
	print("[INFO] Detection cyle 1 of 2 for {}.png completed...".format(im_name))


	# create text_array for strOptions.
	# remove exact match for 'INVOICE' from text_array so that
	# we can detect 'Invoice number' with highest accuracy.  
	# text_array contains all detections except 'INVOICE'
	text_array = []
	for ((startX, startY, endX, endY), text) in results:
		text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
		Ratio = fuzz.ratio('INVOICE',text)
		if Ratio != 100:
			text_array.append(text)
	#print(text_array)
		
	print("[INFO] Finding matches...")

	# find the detection most similar to 'str2Match'.
	# store a print the most similar detection with acc.
	# could have done with both 'INVOICE #' as well as
	# 'Invoice number'. But works only with 'Invoice number'.
	# still need to use them as padding will differ for both.
	str2Match = "Invoice number"
	strOptions = text_array
	Ratios = process.extract(str2Match,strOptions)
	print("\n[INFO] Printing top matches for 'Invoice number'")
	print(Ratios)
	# best match
	highest = process.extractOne(str2Match,strOptions)
	print("\n[INFO] Printing best match for 'Invoice number'")
	print("'{}' detected with higest acc of {}%".format(highest[0], highest[1]))

	# Remove all results except 'best match'
	for ((startX, startY, endX, endY), text) in results:
		text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
		if text==highest[0]:
			best_match = ((startX, startY, endX, endY), text)

	print("\n[INFO] Detection cyle 2 of 2 for {}.png started...".format(im_name))

	# detect again this time in the region of higest acc
	# with x-padding-right increased in roi for 'Invoice number'
	# and change roi of 'INVOICE #' so that it detects only number  
	startX = best_match[0][0]
	startY = best_match[0][1]
	endX = best_match[0][2]
	endY = best_match[0][3]
	text = best_match[1]
	text4tweak = best_match[1]

	if fuzz.ratio(text,'Invoice number') > fuzz.ratio(text,'INVOICE #'):
		dX = int((endX - startX) * 1.00)
		# apply padding to each side of the bounding box, respectively
		startX = max(0, startX)
		endX = min(origW, endX + (dX * 2))

	if fuzz.ratio(text,'Invoice number') < fuzz.ratio(text,'INVOICE #'):
		# in order to obtain a better OCR of the text we can potentially
		# apply a bit of padding surrounding the bounding box -- here we
		# are computing the deltas in both the x and y directions
		dX = int((endX - startX) * 0.05)
		dY = int((endY - startY) * 1.00)
		# apply padding to each side of the bounding box, respectively
		startX = max(0, startX - dX)
		startY = max(0, startY + 1*dY)
		endX = min(origW, endX + (dX * 1))
		endY = min(origH, endY + (dY * 2))

	# extract the actual padded ROI
	roi = orig[startY:endY, startX:endX]

	# detect for second time with perfection
	config = ("-l eng --oem 1 --psm 7")
	text = pytesseract.image_to_string(roi, config=config)
	textcopy = text

	print("[INFO] Detection cyle 2 of 2 for {}.png completed...".format(im_name))

	# tweak for faster computation
	if fuzz.ratio(text4tweak,'Invoice number') < fuzz.ratio(text4tweak,'INVOICE #'):
		text = 'INVOICE# ' + text

	print("\n\nDETECTED TEXT")
	print("========")
	print("{}\n\n".format(text))
	print("[INFO] Press Esc. or any other key to close the window and procced to next image.")

	# DISPLAY
	# using OpenCV, then draw the text and a bounding box surrounding
	# the text region of the input image
	output = orig.copy()
	cv2.rectangle(output, (startX, startY), (endX, endY),
		(0, 0, 255), 2)
	cv2.putText(output, text, (startX, startY - 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	# show the output image
	cv2.imshow("Text Detection", output)
	cv2.imwrite("detected_images/{}.png".format(im_name), output)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	if fuzz.ratio(text4tweak,'Invoice number') < fuzz.ratio(text4tweak,'INVOICE #'):
		num = textcopy
	else:
		num = text.split()
		num = str(num[-2]) + " " + str(num[-1])
	detection_details.append(num)

	content.append(detection_details)
	

print(content)


# WRITE TO EXCEL
workbook = xlsxwriter.Workbook('excel_output/OUTPUT_EXCEL_FILE.xlsx') 
worksheet = workbook.add_worksheet("Invoice Numbers")
# Start from the first cell. Rows and 
# columns are zero indexed. 
row = 0
col = 0


for name, invoice_number in (content): 
    worksheet.write(row, col, name) 
    worksheet.write(row, col + 1, invoice_number)
    row += 1
  
workbook.close()

print("="*100)
print("\nOutput Has Been Saved To excel_output/OUTPUT_EXCEL_FILE.xlsx.\n\
Images With Bounding Boxes Have Been Saved In 'detected_images' folder\n")
print("="*100)