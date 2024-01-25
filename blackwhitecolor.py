from flask import Flask, request, redirect, render_template
import numpy as np
import cv2, base64
import io
prototxt = "static/models/colorization_deploy_v2.prototxt"
model = "static/models/colorization_release_v2.caffemodel"
points = "static/models/pts_in_hull.npy"

net = cv2.dnn.readNetFromCaffe(prototxt, model)
pts = np.load(points)

class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]