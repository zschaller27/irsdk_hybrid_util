import numpy as np
import os

"""
Track:
Description: The purpose of this class is to hold and learn the track layout.
             This will allow for hybrid deployment calculations based on straight
             values found.
"""
class Track:
    def __init__(self, given_track="test"):
        self.track = given_track
        self.lap_count = 0
        self.inconsistant_laps = 0

        # Fractional Straight Lengths (holds the length of each full throttle 
        # section as a fraction of a full lap)
        #           Each value in this list holds:
        #               Index 0: start of straight
        #               Index 1: length fraction
        #               Index 2: entry speed
        self.frac_straight_lens = np.array([[]])

        self.load_data()

    def shutdown(self):
        self.save_data()

    ## Accessors ##
    def get_track(self):
        return self.track

    def get_lap_count(self):
        return self.lap_count

    def get_straight_fractions(self):
        return self.frac_straight_lens

    def save_data(self):
        if self.lap_count >= 2:
            np.savetxt("Data/track_data_%s.dat" %self.track.replace(' ', '_'), self.frac_straight_lens)

    ## Modifiers ##
    def load_data(self):
        if os.path.isfile("Data/track_data_%s.dat" %self.track.replace(' ', '_')):
            self.frac_straight_lens = np.array([np.loadtxt("Data/track_data_%s.dat" %self.track.replace(' ', '_'))])
            self.lap_count = 2
            self.print_fracs()
    
    def normailze_fracs(self):
        straight_lengths_sum = np.sum(self.frac_straight_lens[:, 1])
        self.frac_straight_lens[:, 1] = self.frac_straight_lens[:, 1] / straight_lengths_sum

    def new_lap(self, full_throttle_pcts):
        # Check if there is no existing data or more laps have been done with a different number of straights
        if self.lap_count == 0 or self.inconsistant_laps > self.lap_count:
            # Initialize the list to 2D list where every value is 0
            self.frac_straight_lens = np.array(full_throttle_pcts)
            self.lap_count = 0
        # Check if the lap has inconsitant number of straights
        elif self.lap_count <= 2 and len(full_throttle_pcts) != self.frac_straight_lens.shape[0]:
            self.inconsistant_laps += 1
        else:
            self.frac_straight_lens += np.array(full_throttle_pcts)
            self.frac_straight_lens = self.frac_straight_lens / 2
        
        self.normailze_fracs()
        self.lap_count += 1
    
    ## Testing Functions ##
    def print_fracs(self):
        print("After %d Laps:" %self.lap_count)
        for i, straight in enumerate(self.frac_straight_lens):
            print("\tStraight %d:" %i)
            print("\t\tStart: %1.5f" %straight[0])
            print("\t\tLength: %1.5f" %straight[1])
            print("\t\tEntry Speed: %1.5f" %straight[2])
