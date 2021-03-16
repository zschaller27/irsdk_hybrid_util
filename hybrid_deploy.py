
import numpy as np
from hybrid_track import Track

class Deploy_Hybrid:
    def __init__(self, given_track):
        if not isinstance(given_track, Track):
            raise SyntaxError

        self.track = given_track
        self.straight_epsilon = 0.05
        self.pos_epsilon = 0.01

        self.total_straight_pct = 0
        self.straight_deploy_amounts = np.zeros(self.track.get_straight_fractions().shape[0])

    def update(self):
        self.straight_deploy_amounts = np.zeros(self.track.get_straight_fractions().shape[0])

        # Find Amount of Lap that on a Straight
        applicable_straights = np.where(self.track.get_straight_fractions()[:, 1] > self.straight_epsilon)[0]

        # Total Percentage of Lap on Applicable Straight
        total_percentage = np.sum(self.track.get_straight_fractions()[applicable_straights, 1])

        self.straight_deploy_amounts[applicable_straights] = 100 * (self.track.get_straight_fractions()[applicable_straights, 1] / total_percentage)

    def check_deploy(self, pos, allotment):
        # print(pos)
        # print(allotment)
        # print(self.track.get_straight_fractions())
        # Check if on a straight
        for i, straight in enumerate(self.track.get_straight_fractions()):
            if straight[0] <= pos <= straight[0] + straight[1]:
                
                total_deploy = allotment + self.straight_deploy_amounts[i]

                print("DETECTED STRAIGHT:")
                print("\tDriver on Straight %d" %i)
                print("\tDeploy up to %3.1f allotment" %total_deploy)

    def straight_heuristic(straight, total_pct):
        return 100 * (straight[1] / total_pct)
