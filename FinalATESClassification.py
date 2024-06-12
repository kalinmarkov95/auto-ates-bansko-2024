# -*- coding: utf-8 -*-

"""
***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessing,
                       QgsRasterBandStats,
                       QgsFeatureSink,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterString,
                       QgsProcessingParameterField,
                       QgsProcessingParameterRasterDestination,
                       QgsProcessingParameterFeatureSink,
                       QgsRasterLayer,
                       QgsVectorLayer,
                       QgsMessageLog,
                       QgsProcessingParameterFile,
                       Qgis,
                       QgsPathResolver)
from qgis.analysis import QgsRasterCalculator, QgsRasterCalculatorEntry
from qgis import processing
import numpy
import glob
import os
import math
import rasterio, rasterio.mask
from rasterio.fill import fillnodata
from osgeo import gdal
from datetime import datetime
from datetime import date
import pathlib
import sys
import shutil
import subprocess
from skimage import morphology

class AutomizedATESAlgorithm(QgsProcessingAlgorithm):

    INPUTFLUXLAYER = 'INPUTFLUXLAYER'
    INPUTSLOPELAYER = 'INPUTSLOPELAYER'
    INPUTFORESTDENSITYLAYER = 'INPUTFORESTDENSITYLAYER'
    FOLDERFORINTERMEDIATEPROCESSING = 'FOLDERFORINTERMEDIATEPROCESSING'
    OUTPUTFOLDER = 'OUTPUTFOLDER'

    def tr(self, string):
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return AutomizedATESAlgorithm()

    def name(self):
        
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'AutoATES Bansko - perform the combination steps to get ATES'


    def displayName(self):
        
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr('AutoATES Bansko - perform the combination steps to get ATES')


    def group(self):
        
        """
        Returns the name of the group this algorithm belongs to. This string
        should be localised.
        """
        return self.tr('AutoATES Scripts')


    def groupId(self):
        
        """
        Returns the unique ID of the group this algorithm belongs to. This
        string should be fixed for the algorithm, and must not be localised.
        The group id should be unique within each provider. Group id should
        contain lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'AutoATES Scripts'


    def shortHelpString(self):
        
        """
        Returns a localised short helper string for the algorithm. This string
        should provide a basic description about what the algorithm does and the
        parameters and outputs associated with it..
        """
        return self.tr("This algorithm performs the final classification procedure to get the resulting ATES, based on the algorithm used for Bansko ATES 2024. In order for this algorithm to work, you must have SAGANG and GDAL installed in QGIS. Also, the three input layers must have EXACTLY the same width and height (number of rows and cols) and the same pixel size - so they must match 100% spatially in order for this algorithm to work! Otherwise, weird behavior can be expected!")


    def initAlgorithm(self, config=None):
        
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUTFLUXLAYER,
                self.tr('Input flux layer')
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUTSLOPELAYER,
                self.tr('Input slope layer')
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUTFORESTDENSITYLAYER,
                self.tr('Input forest density layer'),
            )
        )
                
        self.addParameter(
            QgsProcessingParameterFile(
                self.FOLDERFORINTERMEDIATEPROCESSING,
                self.tr("Processing folder - a directory with the current date will get created within this folder, and it will hold all the intermiediate files"),
                behavior=QgsProcessingParameterFile.Folder
                )
        )
        
        self.addParameter(
            QgsProcessingParameterFile(
                self.OUTPUTFOLDER,
                self.tr("Output folder - the final result, called FinalATES.tif, will get placed within this folder"),
                behavior=QgsProcessingParameterFile.Folder
                )
        )


    def createWorkingDirectory(self, baseWorkingDir):
        
        current_date = date.today()
        current_date_and_time = str(current_date) + "-" + datetime.now().strftime("%H:%M:%S").replace(":","")
        
        fullBaseWorkingDir = os.path.join(baseWorkingDir, "AutoATESBanskoFinalATESCombination" + current_date_and_time)
        os.mkdir(fullBaseWorkingDir)
        
        return fullBaseWorkingDir
        

    def categorizeFluxIntoATESClasses(self, fluxCategorized, outputlocation):
        
        outputFile = os.path.join(outputlocation, "FluxCategorizedIntoATESClasses.tif")
        entries = []
        
        fluxCategorizedLayerBand = QgsRasterCalculatorEntry()
        fluxCategorizedLayerBand.ref = 'fluxCategorizedLayerBand@1'
        fluxCategorizedLayerBand.raster = fluxCategorized
        fluxCategorizedLayerBand.bandNumber = 1
        entries.append(fluxCategorizedLayerBand)
        
        calculation = QgsRasterCalculator(' if(fluxCategorizedLayerBand@1 < 0.03, 1, if(fluxCategorizedLayerBand@1 < 0.35, 2, 3))', outputFile, 'GTiff', fluxCategorized.extent(), fluxCategorized.width(), fluxCategorized.height(), entries)
        calculation.processCalculation()
                
        return outputFile
    

    def categorizeSlopeBinary45DegreesAndAbove(self, slope, outputlocation):
        
        outputFile = os.path.join(outputlocation, "SlopeBinary45DegreesAndAbove.tif")
        entries = []
        
        slopeCategorizedLayerBand = QgsRasterCalculatorEntry()
        slopeCategorizedLayerBand.ref = 'slopeCategorizedLayerBand@1'
        slopeCategorizedLayerBand.raster = slope
        slopeCategorizedLayerBand.bandNumber = 1
        entries.append(slopeCategorizedLayerBand)
        
        calculation = QgsRasterCalculator(' if(slopeCategorizedLayerBand@1 < 45, 0, 1)', outputFile, 'GTiff', slope.extent(), slope.width(), slope.height(), entries)
        calculation.processCalculation()
                
        return outputFile
        

    def categorizeForestDensityBinaryBelow25Percent(self, forestDensity, outputlocation):
        
        outputFile = os.path.join(outputlocation, "ForestDensityBinaryBelow25Percent.tif")
        entries = []
        
        forestDensityLayerCategorizedBand = QgsRasterCalculatorEntry()
        forestDensityLayerCategorizedBand.ref = 'forestDensityLayerCategorizedBand@1'
        forestDensityLayerCategorizedBand.raster = forestDensity
        forestDensityLayerCategorizedBand.bandNumber = 1
        entries.append(forestDensityLayerCategorizedBand)
        
        calculation = QgsRasterCalculator('if(forestDensityLayerCategorizedBand@1 < 25, 1, 0)', outputFile, 'GTiff', forestDensity.extent(), forestDensity.width(), forestDensity.height(), entries)
        calculation.processCalculation()
                
        return outputFile
    
    
    def combineLayers(self, slopeBinary, forestDensityBinary, originalATESClassification):
        
        if slopeBinary == 1 and forestDensityBinary == 1:
            return 3
        else:
            return originalATESClassification
    

    def find_longest_border_value(self, data, lab, num_labels, valueThatWillBeCastToNeighboring):
        
        rows, cols = data.shape
        result = data.copy()
        numberOfZonesSmoothedOut = 0
        
        # Iterate through each no-data region
        for region_num in range(1, num_labels + 1):
            
            # Find the coordinates of the current no-data region
            region_coords = numpy.where(lab == region_num)
            neighbor_counts = {}
            considered_indices = set()

            # Iterate through each pixel in the no-data region
            for (i, j) in zip(*region_coords):
                
                if data[i, j] == valueThatWillBeCastToNeighboring:
                    
                    numberOfZonesSmoothedOut = numberOfZonesSmoothedOut + 1
                    # Find neighbors
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if (ni, nj) not in considered_indices:
                                if 0 <= ni < rows and 0 <= nj < cols:
                                    if data[ni, nj] != valueThatWillBeCastToNeighboring:
                                        region_value = data[ni, nj]
                                        if region_value in neighbor_counts:
                                            neighbor_counts[region_value] += 1
                                        else:
                                            neighbor_counts[region_value] = 1
                                considered_indices.add((ni, nj))
            
            # Find the region with the maximum shared border
            if neighbor_counts:
                dominant_region = max(neighbor_counts, key=neighbor_counts.get)
                result[region_coords] = dominant_region

        QgsMessageLog.logMessage("Number of zones smoothed out:" + str(numberOfZonesSmoothedOut), "AutoATES Bansko 2024")
        return result, numberOfZonesSmoothedOut
        
   
    def processAlgorithm(self, parameters, context, feedback):
        
        workingDirectory = self.createWorkingDirectory(parameters['FOLDERFORINTERMEDIATEPROCESSING'])
        fluxTestLayer = self.parameterAsRasterLayer(parameters, self.INPUTFLUXLAYER, context)
        slopeLayer = self.parameterAsRasterLayer(parameters, self.INPUTSLOPELAYER, context)
        forestDensityLayer = self.parameterAsRasterLayer(parameters, self.INPUTFORESTDENSITYLAYER, context)
        
        fluxCategorizedIntoATESClasses = self.categorizeFluxIntoATESClasses(fluxTestLayer, workingDirectory)
        slopeBinaryClassified45DegreesAndAbove = self.categorizeSlopeBinary45DegreesAndAbove(slopeLayer, workingDirectory)
        forestDensityLayerBinaryClassifiedBelow25Percent = self.categorizeForestDensityBinaryBelow25Percent(forestDensityLayer, workingDirectory)
        
        fluxCategorizedIntoATESClassesGdal = gdal.Open(fluxCategorizedIntoATESClasses)
        originalATESClassesArray = numpy.array(fluxCategorizedIntoATESClassesGdal.GetRasterBand(1).ReadAsArray())
        
        slopeBinaryClassified45DegreesAndAboveGdal = gdal.Open(slopeBinaryClassified45DegreesAndAbove)
        slopeBinaryArray = numpy.array(slopeBinaryClassified45DegreesAndAboveGdal.GetRasterBand(1).ReadAsArray())
                
        forestDensityLayerBinaryClassifiedBelow25PercentGdal = gdal.Open(forestDensityLayerBinaryClassifiedBelow25Percent)
        forestDensityBinaryArray = numpy.array(forestDensityLayerBinaryClassifiedBelow25PercentGdal.GetRasterBand(1).ReadAsArray())
        
        QgsMessageLog.logMessage("slope array shape is:" + str(slopeBinaryArray.shape), "AutoATES Bansko 2025")
        QgsMessageLog.logMessage("forest array shape is:" + str(forestDensityBinaryArray.shape), "AutoATES Bansko 2025")
        QgsMessageLog.logMessage("originalATESClassesArray array shape is:" + str(originalATESClassesArray.shape), "AutoATES Bansko 2025")
        
        combined_array = numpy.zeros_like(originalATESClassesArray)

        for i in range(combined_array.shape[0]):
            for j in range(combined_array.shape[1]):
                combined_array[i, j] = self.combineLayers(slopeBinaryArray[i, j], forestDensityBinaryArray[i, j], originalATESClassesArray[i, j])
                
        output_file = os.path.join(workingDirectory, "ATESClassesCombined.tif")
        
        with rasterio.open(fluxCategorizedIntoATESClasses) as src:
            profile = src.profile
            with rasterio.open(output_file, 'w', **profile) as dst:
                dst.write(combined_array, 1)
        
        # Smoothing procedure
        islandFilterSizeSimple = 3000
        islandFilterSizeChallenging = 2000
        islandFilterSizeComplex = 1000
        
        geoTransform = fluxCategorizedIntoATESClassesGdal.GetGeoTransform()
        pixelSizeX = geoTransform[1]
        pixelSizeY =-geoTransform[5]
        
        numCellsSimple = numpy.around(islandFilterSizeSimple / (pixelSizeX * pixelSizeY))
        numCellsChallenging = numpy.around(islandFilterSizeChallenging / (pixelSizeX * pixelSizeY))
        numCellsComplex = numpy.around(islandFilterSizeComplex / (pixelSizeX * pixelSizeY))
        
        originalATESClasses = rasterio.open(output_file)
        originalATESClasses = originalATESClasses.read(1)
        originalATESClasses = originalATESClasses.reshape(1, originalATESClasses.shape[0], originalATESClasses.shape[1])
        
        labelledArray, num_labels = morphology.label(originalATESClasses, connectivity=2, return_num=True)
        result = {'labeled_array': labelledArray, 'original_values': originalATESClasses}

        # Get the labeled array and original values from the result dictionary
        lab = result['labeled_array']
        values = result['original_values']
       
        rg = numpy.arange(1, num_labels+1, 1)

        for i in rg:
            
            occurrences = numpy.count_nonzero(lab == i)            
            indices = numpy.where(lab == i)
            clusterValues = values[indices]
            atesClassOfCluster = clusterValues[0]
            
            if atesClassOfCluster == 1:
                if occurrences < numCellsSimple:
                    values[indices] = 0
            elif atesClassOfCluster == 2:
                if occurrences < numCellsChallenging:
                    values[indices] = 0
            else:
                if occurrences < numCellsComplex:
                    values[indices] = 0

        lab = lab.reshape(lab.shape[1], lab.shape[2])
        values = values.reshape(values.shape[1], values.shape[2])
        
        output_file = os.path.join(workingDirectory, "Lab.tif")
        
        with rasterio.open(fluxCategorizedIntoATESClasses) as src:
            profile = src.profile
            with rasterio.open(output_file, 'w', **profile) as dst:
                dst.write(lab, 1)
        
        output_file = os.path.join(workingDirectory, "Values.tif")
        
        with rasterio.open(fluxCategorizedIntoATESClasses) as src:
            profile = src.profile
            with rasterio.open(output_file, 'w', **profile) as dst:
                dst.write(values, 1)
        
        while True:
            
            QgsMessageLog.logMessage("In the loop", "AutoATES Bansko 2024")
            values, numberOfZonesSmoothedOut = self.find_longest_border_value(values, lab, num_labels, 0)
            if numberOfZonesSmoothedOut == 0:
                break
                
        output_file = os.path.join(parameters["OUTPUTFOLDER"], "FinalATES.tif")
        with rasterio.open(fluxCategorizedIntoATESClasses) as src:
            profile = src.profile
            profile.update({"driver": "GTiff", "nodata": -9999, 'dtype': 'int16'})
            with rasterio.open(output_file, 'w', **profile) as dst:
                dst.write(values, 1)

        return {}
        