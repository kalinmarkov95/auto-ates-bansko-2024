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

class PraAndForestProtectionLayerCalculation(QgsProcessingAlgorithm):

    INPUTDEMLAYER = 'INPUTDEMLAYER'
    INPUTFORESTDENSITYLAYER = 'INPUTFORESTDENSITYLAYER'
    FOLDERFORINTERMEDIATEPROCESSING = 'FOLDERFORINTERMEDIATEPROCESSING'
    #FLOWPYBASEDIR = 'FLOWPYBASEDIR'
    OUTPUTFOLDER = 'OUTPUTFOLDER'

    def tr(self, string):
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return PraAndForestProtectionLayerCalculation()

    def name(self):
        
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'AutoATES Bansko - PRA and forest protection layer calculation'


    def displayName(self):
        
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr('AutoATES Bansko - PRA and forest protection layer calculation')


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
        return self.tr("This algorithm calculates the PRA and forest protection layers, according to the Bansko ATES 2024 algorithm. After that, these two layers can be manually passed in to FlowPy. In order for this algorithm to work, you must have SAGANG, gdal, and the WhiteboxTools plugin installed in QGIS.")


    def initAlgorithm(self, config=None):
        
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUTDEMLAYER,
                self.tr('Input DEM layer')
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
        
        """
        self.addParameter(
            QgsProcessingParameterFile(
                self.FLOWPYBASEDIR,
                self.tr("FlowPy base dir"),
                behavior=QgsProcessingParameterFile.Folder
                )
        )
        """
                
        self.addParameter(
            QgsProcessingParameterFile(
                self.OUTPUTFOLDER,
                self.tr("Output folder - the reclassified PRA raster and the forest protection layer will be placed here"),
                behavior=QgsProcessingParameterFile.Folder
                )
        )


    def createWorkingDirectory(self, baseWorkingDir):
        
        current_date = date.today()
        current_date_and_time = str(current_date) + "-" + datetime.now().strftime("%H:%M:%S").replace(":","")
        
        fullBaseWorkingDir = os.path.join(baseWorkingDir, "AutoATESBanskoPRAAndForestProtectionCalculation" + current_date_and_time)
        os.mkdir(fullBaseWorkingDir)
 
        return fullBaseWorkingDir
        

    def calculateSlope(self, dem, outputlocation):
        
        outputFile = os.path.join(outputlocation, "Slope.tif")
        parameters = {'INPUT': dem,
                      'OUTPUT': outputFile}
        processing.run('qgis:slope', parameters)
        return outputFile


    def calculatePlanCurvature(self, dem, outputlocation):
        
        outputFile = os.path.join(outputlocation, "PlanCurvature.tif")
        parameters = {'dem': dem,
                      'output': outputFile}
        processing.run('wbt:PlanCurvature', parameters)
        return outputFile
    

    # ArcGIS outputs the plan curvature value in meters elevation change (up or down) per 100 meters
    # So a value of 50 means that in those 100 meters, the surface curves upwards 50 meters (measured vertically)
    # The tool we use here, Whitebox Tools, outputs this measurement per meter. It would output 0.5 in this case, 
    # indicating we have a 0.5 meter upwards curve per 1 meter. Negative values indicate downwards curving.
    # We need to scale by multiplying by 100 so that we have easier-to-work with numbers.
    def scalePlanCurvature(self, planCurvature, outputlocation):
        
        outputFile = os.path.join(outputlocation, "PlanCurvatureScaledBy100.tif")
        
        planCurvatureLayer = QgsRasterLayer(planCurvature)
        entries = []
        
        planCurvatureLayerBand = QgsRasterCalculatorEntry()
        planCurvatureLayerBand.ref = 'planCurvatureLayerBand@1'
        planCurvatureLayerBand.raster = planCurvatureLayer
        planCurvatureLayerBand.bandNumber = 1
        entries.append(planCurvatureLayerBand)
                
        calculation = QgsRasterCalculator('planCurvatureLayerBand@1 * 100', outputFile, 'GTiff', planCurvatureLayer.extent(), planCurvatureLayer.width(), planCurvatureLayer.height(), entries)
        calculation.processCalculation()
        
        return outputFile
    
    
    def makeSlopeIn0To1Interval(self, slope, outputlocation):
        
        outputFile = os.path.join(outputlocation, "SlopeInterval0To1.tif")
        
        slopeLayer = QgsRasterLayer(slope)
        entries = []
        
        slopeLayerBand = QgsRasterCalculatorEntry()
        slopeLayerBand.ref = 'slopeLayerBand@1'
        slopeLayerBand.raster = slopeLayer
        slopeLayerBand.bandNumber = 1
        entries.append(slopeLayerBand)
         
        calculation = QgsRasterCalculator('(1.0) / ( 1.0 + ((slopeLayerBand@1 - 43.0) / 11.0)  ^  8)', outputFile, 'GTiff', slopeLayer.extent(), slopeLayer.width(), slopeLayer.height(), entries)
        calculation.processCalculation()
        
        return outputFile
        

    def makePlanCurvatureIn0To1Interval(self, planCurvature, outputlocation):
        
        outputFile = os.path.join(outputlocation, "PlanCurvatureInterval0To1.tif")
        entries = []
        
        planCurvatureLayer = QgsRasterLayer(planCurvature)
        planCurvatureLayerBand = QgsRasterCalculatorEntry()
        planCurvatureLayerBand.ref = 'planCurvatureLayerBand@1'
        planCurvatureLayerBand.raster = planCurvatureLayer
        planCurvatureLayerBand.bandNumber = 1
        entries.append(planCurvatureLayerBand)
        
        max = self.getMaximumValueInRaster(planCurvature)
        expression = 'if(planCurvatureLayerBand@1 < 0, 1, (-1/' + str(max) + ') * planCurvatureLayerBand@1 + 1)'
                        
        calculation = QgsRasterCalculator(expression, outputFile, 'GTiff', planCurvatureLayer.extent(), planCurvatureLayer.width(), planCurvatureLayer.height(), entries)
        calculation.processCalculation()
                
        return outputFile
        

    def makeForestDensityIn0To1Interval(self, forestDensityLayer, outputlocation):
        
        outputFile = os.path.join(outputlocation, "ForestDensityInterval0To1.tif")
        entries = []
        
        forestDensityLayerBand = QgsRasterCalculatorEntry()
        forestDensityLayerBand.ref = 'forestDensityLayerBand@1'
        forestDensityLayerBand.raster = forestDensityLayer
        forestDensityLayerBand.bandNumber = 1
        entries.append(forestDensityLayerBand)
                
        calculation = QgsRasterCalculator('(1.0) / ( 1.0 + ((forestDensityLayerBand@1 + 15.0) / 40.0)  ^  7)', outputFile, 'GTiff', forestDensityLayer.extent(), forestDensityLayer.width(), forestDensityLayer.height(), entries)
        calculation.processCalculation()
                
        return outputFile
    

    def getMaximumValueInRaster(self, pathToRaster):
        
        raster = gdal.Open(pathToRaster)
        data_band = raster.GetRasterBand(1)
        data = numpy.array(data_band.ReadAsArray())
        return data.max()
        

    def calculateMinimumBetweenRasters(self, slope, planCurvature, forestDensity, outputlocation):
        
        outputFile = os.path.join(outputlocation, "MinParam.tif")
        parameters = {'INPUT': [slope, planCurvature, forestDensity],
                      'STATISTIC': 6,
                      'REFERENCE_LAYER': slope,
                      'OUTPUT': outputFile}
                      
        processing.run('native:cellstatistics', parameters)
        return outputFile
    
    
    def calculateYParam(self, minParam, outputlocation):
        
        outputFile = os.path.join(outputlocation, "YParam.tif")
        entries = []
        
        minParamLayer = QgsRasterLayer(minParam)
        minParamLayerBand = QgsRasterCalculatorEntry()
        minParamLayerBand.ref = 'minParamLayerBand@1'
        minParamLayerBand.raster = minParamLayer
        minParamLayerBand.bandNumber = 1
        entries.append(minParamLayerBand)
        
        calculation = QgsRasterCalculator('1 - minParamLayerBand@1', outputFile, 'GTiff', minParamLayer.extent(), minParamLayer.width(), minParamLayer.height(), entries)
        calculation.processCalculation()
                
        return outputFile
        
        
    def calculatePRA(self, slope, planCurvature, forestDensity, minParam, yParam, outputlocation):
        
        outputFile = os.path.join(outputlocation, "PRA.tif")
        entries = []
        
        slopeLayer = QgsRasterLayer(slope)
        slopeLayerBand = QgsRasterCalculatorEntry()
        slopeLayerBand.ref = 'slopeLayerBand@1'
        slopeLayerBand.raster = slopeLayer
        slopeLayerBand.bandNumber = 1
        entries.append(slopeLayerBand)
        
        
        planCurvatureLayer = QgsRasterLayer(planCurvature)
        planCurvatureLayerBand = QgsRasterCalculatorEntry()
        planCurvatureLayerBand.ref = 'planCurvatureLayerBand@1'
        planCurvatureLayerBand.raster = planCurvatureLayer
        planCurvatureLayerBand.bandNumber = 1
        entries.append(planCurvatureLayerBand)
        
        
        forestDensityLayer = QgsRasterLayer(forestDensity)
        forestDensityLayerBand = QgsRasterCalculatorEntry()
        forestDensityLayerBand.ref = 'forestDensityLayerBand@1'
        forestDensityLayerBand.raster = forestDensityLayer
        forestDensityLayerBand.bandNumber = 1
        entries.append(forestDensityLayerBand)
        
        
        minParamLayer = QgsRasterLayer(minParam)
        minParamLayerBand = QgsRasterCalculatorEntry()
        minParamLayerBand.ref = 'minParamLayerBand@1'
        minParamLayerBand.raster = minParamLayer
        minParamLayerBand.bandNumber = 1
        entries.append(minParamLayerBand)
        
        
        yParamLayer = QgsRasterLayer(yParam)
        yParamLayerBand = QgsRasterCalculatorEntry()
        yParamLayerBand.ref = 'yParamLayerBand@1'
        yParamLayerBand.raster = yParamLayer
        yParamLayerBand.bandNumber = 1
        entries.append(yParamLayerBand)
        
        calculation = QgsRasterCalculator('yParamLayerBand@1 * minParamLayerBand@1 + (((1 - yParamLayerBand@1) * (slopeLayerBand@1 + planCurvatureLayerBand@1 + forestDensityLayerBand@1)) / 3.0)', outputFile, 'GTiff', slopeLayer.extent(), slopeLayer.width(), slopeLayer.height(), entries)
        calculation.processCalculation()
                
        return outputFile
        

    def reclassifyPRA(self, PRA, threshold, outputlocation):
        
        outputFile = os.path.join(outputlocation, "PRAReclassThreshold03.tif")
        entries = []
        
        praLayer = QgsRasterLayer(PRA)
        praLayerBand = QgsRasterCalculatorEntry()
        praLayerBand.ref = 'praLayerBand@1'
        praLayerBand.raster = praLayer
        praLayerBand.bandNumber = 1
        entries.append(praLayerBand)
        
        expression = 'if(praLayerBand@1 > ' + str(threshold) + ', 1, 0)'
        calculation = QgsRasterCalculator(expression, outputFile, 'GTiff', praLayer.extent(), praLayer.width(), praLayer.height(), entries)
        calculation.processCalculation()
                
        return outputFile
        

    def calculateForestProtection(self, forestDensityLayerInterval0To1, outputlocation):
        
        outputFile = os.path.join(outputlocation, "ForestProtection.tif")
        entries = []
        
        forestDensityLayer = QgsRasterLayer(forestDensityLayerInterval0To1)
        forestDensityLayerBand = QgsRasterCalculatorEntry()
        forestDensityLayerBand.ref = 'forestDensityLayerBand@1'
        forestDensityLayerBand.raster = forestDensityLayer
        forestDensityLayerBand.bandNumber = 1
        entries.append(forestDensityLayerBand)
        
        calculation = QgsRasterCalculator('1 - forestDensityLayerBand@1', outputFile, 'GTiff', forestDensityLayer.extent(), forestDensityLayer.width(), forestDensityLayer.height(), entries)
        calculation.processCalculation()
                
        return outputFile
        
    def processAlgorithm(self, parameters, context, feedback):
        
        workingDirectory = self.createWorkingDirectory(parameters['FOLDERFORINTERMEDIATEPROCESSING'])
       
        DEM = parameters["INPUTDEMLAYER"]
        slope = self.calculateSlope(DEM, workingDirectory)
        planCurvature = self.calculatePlanCurvature(DEM, workingDirectory)
        forestDensityLayer = self.parameterAsRasterLayer(parameters, self.INPUTFORESTDENSITYLAYER, context)
                
        slopeInterval0To1 = self.makeSlopeIn0To1Interval(slope, workingDirectory)
        planCurvatureScaled = self.scalePlanCurvature(planCurvature, workingDirectory)
        planCurvatureInterval0To1 = self.makePlanCurvatureIn0To1Interval(planCurvatureScaled, workingDirectory)
        forestDensityInterval0To1 = self.makeForestDensityIn0To1Interval(forestDensityLayer, workingDirectory)
        
        minParam = self.calculateMinimumBetweenRasters(slopeInterval0To1, planCurvatureInterval0To1, forestDensityInterval0To1, workingDirectory)
        yParam = self.calculateYParam(minParam, workingDirectory)
        pra = self.calculatePRA(slopeInterval0To1, planCurvatureInterval0To1, forestDensityInterval0To1, minParam, yParam, workingDirectory)

        outputFolder = parameters['OUTPUTFOLDER']
        praReclassThreshold = 0.3

        praReclass = self.reclassifyPRA(pra, praReclassThreshold, outputFolder)
        forestProtection = self.calculateForestProtection(forestDensityInterval0To1, outputFolder)
        
        
        # Calling FlowPy from this script currently does not work - needs to be implemented in the future
        """
        flowPyBaseDir = parameters["FLOWPYBASEDIR"]
        pathToFlowPyMainFile = os.path.join(parameters["FLOWPYBASEDIR"], "main.py")
        demLayer = self.parameterAsRasterLayer(parameters, self.INPUTDEMLAYER, context)
        data_provider = demLayer.dataProvider()
        demFilePath = data_provider.dataSourceUri()
        demFilePath = demFilePath.split('|')[0]
        
        command = f'python {pathToFlowPyMainFile} 23 8 {workingDirRunouts} {demFilePath} {praReclass} forest={forestProtection} flux=0.003 max_z=8849'
            
        feedback.pushInfo(f'Running command: {command}')
        
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, check=True
            )
            feedback.pushInfo(f'Subprocess output: {result.stdout}')
        except subprocess.CalledProcessError as e:
            feedback.reportError(f'Subprocess failed with error: {e.stderr}', fatalError=True)
            raise e
        """
        return {}
        