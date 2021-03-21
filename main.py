import csv
import math
import plotly.express as px
import pandas as pd
import numpy as np
import random as rand


PLOTLY_API_TOKEN = "sk.eyJ1IjoiZ2Vvcmdld2lnbGV5IiwiYSI6ImNrbWlwbGJpNTBlejAzMG1rODR3ZWF4NzUifQ.GgfusEAZP7oK6CRnvKTG3g"

ONE_DEGREE_Y = 111111    # 111.111km is roughly 1 degree north in longitude, for latitude it is 111,111*cos(lat)
OCEAN_VECTOR_FROM_WIND = 1   # how much wind vectors affect current vectors


class Object:
    currentlat = 0
    currentlon = 0
    mass = 0
    dragFactor = 0
    name = ""

    def __init__(self, lon, lat, drag, name):   # drones will move 20-30% slower than current
        self.currentlon = lon
        self.currentlat = lat
        self.dragFactor = drag  # proportional to density of object and the aerodynamics of the object
        self.name = name

    def ConvertXMetersToLon(self, xmeters):
        return self.currentlon + ( xmeters / (ONE_DEGREE_Y * math.cos((self.currentlat * (math.pi / 180)))))

    def ConvertYMetersToLat(self, ymeters):
        return self.currentlat + (ymeters / ONE_DEGREE_Y)

    def CalculateNextLocation(self, timestep, vectorreference):  # timestep in seconds
        oceancurrentvector = vectorreference.CalculateNextVector(self.currentlat, self.currentlon) # returns wind vector
        oceancurrentvector.u *= OCEAN_VECTOR_FROM_WIND
        oceancurrentvector.v *= OCEAN_VECTOR_FROM_WIND
        # calculate offset in x and y in meters
        xoffset = timestep * oceancurrentvector.u * self.dragFactor
        yoffset = timestep * oceancurrentvector.v * self.dragFactor
        # convert to new lat long
        self.currentlon = self.ConvertXMetersToLon(xoffset)
        self.currentlat = self.ConvertYMetersToLat(yoffset)


class WindVector:
    direction = 0   # in radians from north (0 - 2pi)
    u = 0
    v = 0
    magnitude = 0   # in m/s
    lon = 0     # x
    lat = 0     # y

    def __init__(self, lon, lat, u, v):
        self.lon = lon
        self.lat = lat
        self.u = u
        self.v = v
        self.magnitude = (math.sqrt((u**2)+(v**2)))
        self.direction = math.acos(( math.fabs(v) / self.magnitude))   # top right quadrant (default)
        if v < 0 and u < 0:     # bottom left quadrant
            self.direction += math.pi
        if v < 0 and u > 0:     # bottom right quadrant
            self.direction += (math.pi / 2)
        if v > 0 and u < 0:     # top left quadrant
            self.direction = (math.pi * 2) - self.direction


class VectorField:

    latdata = []    # y
    londata = []    # x
    udata = []    # horizontal vector
    vdata = []    # vertical vector
    vectors = []

    def __init__(self):
        # load lat and lon values into class array attributes
        latvals = open("data/lat.csv", 'r')
        lonvals = open("data/lon.csv", 'r')
        latreader = csv.reader(latvals, delimiter=',')
        lonreader = csv.reader(lonvals, delimiter=',')
        self.latdata = self.LoadLatLonToArray(latreader)
        self.londata = self.LoadLatLonToArray(lonreader)
        latvals.close()
        lonvals.close()

        uvals = open("data/uvals.csv", 'r')
        vvals = open("data/vvals.csv", 'r')
        ureader = csv.reader(uvals, delimiter=',')
        vreader = csv.reader(vvals, delimiter=',')
        self.udata = self.LoadUVToArray(ureader)
        self.vdata = self.LoadUVToArray(vreader)
        uvals.close()
        vvals.close()

        self.CreateVectors()

    def LoadLatLonToArray(self, reader):
        data = []
        for line in reader:
            for value in line:
                data.append(float(value))
        return data

    def LoadUVToArray(self, reader):
        data = []
        x = 0
        y = 0
        for line in reader:
            temp = []
            for value in line:
                temp.append(float(value))
                x += 1
            data.append(temp)
            y += 1
            x = 0
        return data

    def CreateVectors(self):
        for x in range(len(self.londata)):
            temp = []
            for y in range(len(self.latdata)):
                temp.append(WindVector(self.londata[x], self.latdata[y], self.udata[x][y], self.vdata[x][y]))
            self.vectors.append(temp)

    def FindNearest(self, array, value):  # https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def CalculateNextVector(self, olat, olon):
        finalx, finaly = 0, 0
        finalx = self.FindNearest(self.londata, olon)
        finaly = self.FindNearest(self.latdata, olat)

        # take average (ideally billinear interp)
        val = self.vectors[finalx][finaly]
        val.u = (self.vectors[finalx][finaly].u + self.vectors[finalx + 1][finaly].u) / 2
        val.v = (self.vectors[finalx][finaly].v + self.vectors[finalx + 1][finaly].v) / 2
        return val


class SimulationVis:
    # sim data
    objects = []
    vecdata = VectorField()

    # visualization data
    dataframe = []
    listofrows = [] # name, lat, lon, time

    def AddObject(self, lon, lat, dragfactor, name):
        if lon < self.vecdata.londata[0] or lon > self.vecdata.londata[-1]:
            print("ERROR: out of bounds longitude: "+ name + "," + str(lon) + "," + str(lat))
            exit(-1)
        if lat > self.vecdata.latdata[0] or lat < self.vecdata.latdata[-1]:
            print("ERROR: out of bounds latitude: \t"+ name + "\t lon:" + str(lon) + "\t lat:" + str(lat))
            exit(-1)
        self.objects.append(Object(lon, lat, dragfactor, name))

    def Simulate(self, timestep, numberofstep):
        if len(self.objects) == 0:
            print("Add object to simulator first")
            return
        for obj in self.objects:
            for x in range(numberofstep):
                obj.CalculateNextLocation(timestep, self.vecdata)
                self.listofrows.append([obj.name, obj.currentlat, obj.currentlon, (str(round(((timestep*x)/3600), 1)) + " hours")])

    def RenderToMap(self):
        self.dataframe = pd.DataFrame(self.listofrows, columns=["name","lat", "lon", "time"])

        fig = px.line_mapbox(self.dataframe, lat="lat", lon="lon", hover_name="name", hover_data=["lat", "lon", "time"],
                             color="name", title="Ocean Flow Simulation", zoom=3, height=900, width=1600)

        fig.update_layout(mapbox_style="open-street-map", mapbox_zoom=4, mapbox_center_lat=36.9,
                          margin={"r": 0, "t": 0, "l": 0, "b": 0},
                          legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, ))

        fig.show()


sim = SimulationVis()

sim.AddObject(180, 10, 0.8, "Trash Deposit 1")
sim.AddObject(160, 40, 0.8, "Trash Deposit 2")
sim.AddObject(140, 20, 0.8, "Trash Deposit 3")
sim.AddObject(200, 30, 0.8, "Trash Deposit 4")
sim.AddObject(190, 50, 0.8, "Trash Deposit 5")
sim.AddObject(202.15, 21.4, 0.6, "Drone 1")  # drone moves slower than plastic and waste

sim.Simulate(7200, 500)
sim.RenderToMap()