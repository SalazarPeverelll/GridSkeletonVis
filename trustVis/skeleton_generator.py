

import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors



class SkeletonGenerator:
    """SkeletonGenerator except allows for generate skeleton"""
    def __init__(self, data_provider, epoch, interval=25,base_num_samples=100):
        """
        interval: int : layer number of the radius
        """
        self.data_provider = data_provider
        self.epoch = epoch
        self.interval = interval
        self.base_num_samples= base_num_samples
       
    def skeleton_gen(self):
        torch.manual_seed(0)  # freeze the radom seed
        torch.cuda.manual_seed_all(0)

        # Set the random seed for numpy
        np.random.seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        train_data=self.data_provider.train_representation(epoch=self.epoch)
        train_data = torch.Tensor(train_data)
        center = train_data.mean(dim=0)
        # calculate the farest distance
        radius = ((train_data - center)**2).sum(dim=1).max().sqrt()
        # print("radius,radius",radius)

        # min_radius_log = np.log10(1e-3)
        # max_radius_log = np.log10(radius.item())
        # # *****************************************************************************************
        # # generate 100 points in log space 
        # radii_log = np.linspace(max_radius_log, min_radius_log, self.interval)
        # # convert back to linear space
        # radii = 10 ** radii_log

        # generate points in log space
        # generate points in linear space
        radii = self.create_decreasing_array(1e-3,radius.item(), self.interval)
        epsilon = 1e-2
        train_data_distances = ((train_data - center)**2).sum(dim=1).sqrt().cpu().detach().numpy()
        # calculate the number of samples for each radius

        num_samples_per_radius_l = []
        for r in radii:
            close_points_indices = np.where(np.abs(train_data_distances - r) < epsilon)[0]
            close_points = train_data[close_points_indices].cpu().detach().numpy()
            print("len()",r, len(close_points))
            # calculate the log surface area for the current radius
            # convert it back to the original scale
            # calculate the number of samples
            base_num_samples = len(close_points) + 1
            num_samples = int(base_num_samples * r // 2)
            num_samples_per_radius_l.append(num_samples)
        

        # *****************************************************************************************

        # radii = [radius*1.1, radius, radius / 2, radius / 4, radius / 10, 1e-3]  # radii at which to sample points
        # # num_samples_per_radius_l = [500, 500, 500, 500, 500, 500]  # number of samples per radius
        # aaa = 500
        # num_samples_per_radius_l = [aaa, aaa, aaa, aaa, aaa, aaa]  # number of samples per radius
        print("num_samples_per_radius_l",radii)
        print("num_samples_per_radius_l",num_samples_per_radius_l)
        # list to store samples at all radii
        high_bom_samples = []

        for i in range(len(radii)):
            r = radii[i]

            num_samples_per_radius = num_samples_per_radius_l[i]
            # sample points on the sphere with radius r
            samples = torch.randn(num_samples_per_radius, 512)
            samples = samples / samples.norm(dim=1, keepdim=True) * r

            high_bom_samples.append(samples)

            # concatenate samples from all radii
            high_bom = torch.cat(high_bom_samples, dim=0)

            high_bom = high_bom.cpu().detach().numpy()

        print("shape", high_bom.shape)
        
        # calculate the distance of each training point to the center


        # for each radius, find the training data points close to it and add them to the high_bom
        epsilon = 1e-2  # the threshold for considering a point is close to the radius
        for r in radii:
            close_points_indices = np.where(np.abs(train_data_distances - r) < epsilon)[0]
            close_points = train_data[close_points_indices].cpu().detach().numpy()
            high_bom = np.concatenate((high_bom, close_points), axis=0)

      
        return high_bom
    
    def skeleton_gen_union(self):
        torch.manual_seed(0)  # freeze the radom seed
        torch.cuda.manual_seed_all(0)

        # Set the random seed for numpy
        np.random.seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        train_data=self.data_provider.train_representation(epoch=self.epoch)
        train_data = torch.Tensor(train_data)
        center = train_data.mean(dim=0)
        # calculate the farest distance
        radius = ((train_data - center)**2).sum(dim=1).max().sqrt()
        print("radius,radius",radius)

        min_radius_log = np.log10(1e-3)
        max_radius_log = np.log10(radius.item() * 1.1)
        # *****************************************************************************************
        # generate 100 points in log space 
        radii_log = np.linspace(max_radius_log, min_radius_log, self.interval)
        # convert back to linear space
        radii = 10 ** radii_log

    
        # calculate the number of samples for each radius
        num_samples_per_radius_l = []
        for r in radii:
            # calculate the log surface area for the current radius
            # convert it back to the original scale
            # calculate the number of samples
            num_samples = int(self.base_num_samples * (1+r) // 2)
            num_samples_per_radius_l.append(num_samples)
        

        # *****************************************************************************************

        # radii = [radius*1.1, radius, radius / 2, radius / 4, radius / 10, 1e-3]  # radii at which to sample points
        # # num_samples_per_radius_l = [500, 500, 500, 500, 500, 500]  # number of samples per radius
        # aaa = 500
        # num_samples_per_radius_l = [aaa, aaa, aaa, aaa, aaa, aaa]  # number of samples per radius
        print("num_samples_per_radius_l",radii)
        print("num_samples_per_radius_l",num_samples_per_radius_l)
        # list to store samples at all radii
        high_bom_samples = []

        for i in range(len(radii)):
            r = radii[i]

            num_samples_per_radius = num_samples_per_radius_l[i]
            # sample points on the sphere with radius r
            samples = torch.randn(num_samples_per_radius, 512)
            samples = samples / samples.norm(dim=1, keepdim=True) * r

            high_bom_samples.append(samples)

            # concatenate samples from all radii
            high_bom = torch.cat(high_bom_samples, dim=0)

            high_bom = high_bom.cpu().detach().numpy()

        print("shape", high_bom.shape)
        
        # calculate the distance of each training point to the center
        train_data_distances = ((train_data - center)**2).sum(dim=1).sqrt().cpu().detach().numpy()

        # for each radius, find the training data points close to it and add them to the high_bom
        epsilon = 1e-2  # the threshold for considering a point is close to the radius
        for r in radii:
            close_points_indices = np.where(np.abs(train_data_distances - r) < epsilon)[0]
            close_points = train_data[close_points_indices].cpu().detach().numpy()
            high_bom = np.concatenate((high_bom, close_points), axis=0)

      
        return high_bom
    
    def create_decreasing_array(self,min_val, max_val, levels, factor=0.8):
        # Calculate the total range
        range_val = max_val - min_val

        # Create an array with the specified number of levels
        level_indices = np.arange(levels)

        # Apply the factor to the levels
        scaled_levels = factor ** level_indices

        # Scale the values to fit within the range
        scaled_values = scaled_levels * range_val / np.max(scaled_levels)

        # Shift the values to start at the min_val
        final_values = max_val - scaled_values

        return final_values

    