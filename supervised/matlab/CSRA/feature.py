from __future__ import print_function
import numpy as np
# import scipy.io as sio
# import theano
# import theano.tensor as T
# import lasagne
# import sys
import time

def longtailfunc(x):
    # r = -0.001*x+1.001
    r = 1 / (0.1*(x - 1) + 1)
    # r = 1-np.e**(-0.025)+np.e**(-0.025*x)
    # r = 556.12198*(np.e**((-(np.log(x+1e-5)-12.8)**2)/2*(3.6)**2))/(x+1e-5)	#a
    # r = 42.31545*(np.e**((-(np.log(x+1e-5)-10.4)**2)/2*(3.8)**2))/(x+1e-5)	#c
    # if x <= 1:
    #     r = 1
    # else:
    #     r = 0.01
    return r


class FeatureRepresenter:
    """
    Get the feature representation of the data for giving
    as input to Block1
    """

    def __init__(self, num_p, num_q, k_ability=3, k_difficulty=3, num_dimensions=2):
        """
        k_ability - number of buckets for ability
        k_difficulty - number of buckets for difficulty
        num_p - rows in the matrix
        num_q - columns in the matrix
        num_dimensions - 2 for 2D matrix, etc.
        """
        self.k_ability = k_ability
        self.k_difficulty = k_difficulty
        self.num_dimensions = num_dimensions
        self.num_participants = num_p
        self.num_questions = num_q

    def get_average_ability_2d(self):
        """
        Get average ability in 2D case
        """
        abilities = []
        for i in range(0, self.num_participants):
            correct_count = 0
            for j in range(0, self.num_questions):
                if self.current_proposals[j] == self.input_data[i][j]:
                    correct_count += 1
            ability = float(correct_count) / self.num_questions
            abilities.append((ability, i))
        abilities = np.array(abilities)
        return abilities

    def get_average_difficulty_2d(self):
        """
        Get average difficulty in 2D case
        """
        difficulties = []
        for i in range(0, self.num_questions):
            correct_count = 0
            for j in range(0, self.num_participants):
                if self.current_proposals[i] == self.input_data[j][i]:
                    correct_count += 1
            difficulty = 1 - float(correct_count) / self.num_participants
            difficulties.append((difficulty, i))
        difficulties = np.array(difficulties)
        return difficulties

    def get_buckets(self, attribute, k):
        """
        Get k buckets for the attribute list given in ascending order
        """
        attribute = attribute.tolist()
        attribute.sort(key=lambda x: x[0])
        attribute_step_size = len(attribute) / k
        for i in range(0, k):
            current_bucket = attribute[
                i * attribute_step_size:min((i + 1) * attribute_step_size, len(attribute))]
            for j in range(0, len(current_bucket)):
                current_bucket[j][1] = int(current_bucket[j][1])
            yield current_bucket

    def get_ability_in_bucket(self, ability_bucket):
        """
        Get the ability in the bucket ability_bucket
        """
        difficulties = []
        for i in range(0, self.num_questions):
            correct_count = 0
            for j in range(0, len(ability_bucket)):
                if self.current_proposals[i] == self.input_data[ability_bucket[j][1]][i]:
                    correct_count += 1
            difficulty = 1 - float(correct_count) / len(ability_bucket)
            difficulties.append((difficulty, i))
        difficulties = np.array(difficulties)
        return difficulties

    def get_difficulty_in_bucket(self, difficulty_bucket):
        """
        Get the difficulty in the bucket difficulty_bucket
        """
        abilities = []
        for i in range(0, self.num_participants):
            correct_count = 0
            for j in range(0, len(difficulty_bucket)):
                if self.current_proposals[difficulty_bucket[j][1]] == self.input_data[i][difficulty_bucket[j][1]]:
                    correct_count += 1
            ability = float(correct_count) / len(difficulty_bucket)
            abilities.append((ability, i))
        abilities = np.array(abilities)
        return abilities

    def generate_features_2d(self, input_data, current_proposals):
        """
        Generate the feature input given the data and current proposed
        answer sheet
        """
        assert len(np.shape(input_data)) == self.num_dimensions
        assert len(current_proposals) == self.num_questions
        self.input_data = input_data
        self.current_proposals = current_proposals
        self.abilities = self.get_average_ability_2d()
        self.difficulties = self.get_average_difficulty_2d()
        abilities_bucket_wise = []
        difficulties_bucket_wise = []
        for ability_bucket in self.get_buckets(self.abilities[:], self.k_ability):
            difficulties_bucket_wise.append(
                self.get_ability_in_bucket(ability_bucket))
        for difficulty_bucket in self.get_buckets(self.difficulties[:], self.k_difficulty):
            abilities_bucket_wise.append(
                self.get_difficulty_in_bucket(difficulty_bucket))
        self.abilities_bucket_wise = np.array(abilities_bucket_wise)
        self.difficulties_bucket_wise = np.array(difficulties_bucket_wise)

    def get_features_2d_element(self, person, question):
        """
        Get a particular value in the generated feature matrix
        """
        feature_list = []
        feature_list.append(self.abilities[person][0])
        for i in range(0, self.k_difficulty):
            feature_list.append(self.abilities_bucket_wise[i][person][0])
        feature_list.append(self.difficulties[question][0])
        for i in range(0, self.k_ability):
            feature_list.append(self.difficulties_bucket_wise[i][question][0])
        feature_list = np.array(feature_list)
        return feature_list

    def get_features_2d(self):
        """
        Get the feature list by flattring the feature matrix
        """
        features_list = []
        labels_list = []
        for i in range(0, self.num_participants):
            for j in range(0, self.num_questions):
                features_list.append(self.get_features_2d_element(i, j))
                labels_list.append(self.current_proposals[j])
        features_list = np.array(features_list)
        labels_list = np.array(labels_list)
        return features_list, labels_list


class FeatureRepresenter_3D:
    """
    Get the feature representation of the data for giving
    as input to Block1
    """

    def __init__(self, num_p, num_q, num_o, k_ability=2, k_qdifficulty=3, k_odifficulty=3):
        """
        k_ability - number of buckets for ability
        k_difficulty - number of buckets for difficulty
        num_p - rows in the matrix
        num_q - columns in the matrix
        num_dimensions - 2 for 2D matrix, etc.
        """
        self.k_ability = k_ability
        self.k_qdifficulty = k_qdifficulty
        self.k_odifficulty = k_odifficulty
        self.num_dimensions = 3
        self.num_participants = num_p
        self.num_questions = num_q
        self.num_options = num_o

    # def get_acc_value(self):
    #     acc_value = []
    #     num1 = 20
    #     cut1 = 0.5
    #     step1 = (1-cut1)/num1
    #     for k in range(0,num1):
    #         v1 = 1 - k * step1
    #         acc_value.append(v1)
    #
    #     num2 = 50
    #     cut2 = 0.2
    #     step2 = (cut1-cut2)/(num2-num1)
    #     for k in range(0,num2-num1):
    #         v2 = v1 - k * step2
    #         acc_value.append(v2)
    #
    #     num3 = self.num_options
    #     step3 = (cut2 - 0)/(num3 - num2)
    #     for k in range(0,num3-num2):
    #         v3 = v2 - k * step3
    #         acc_value.append(v3)
    #     return acc_value



    def get_average_ability_3d(self):
        """
        Get average ability in 3D case
        """
        # acc_value = self.get_acc_value()
        abilities = []
        for i in range(0, self.num_participants):#16
            acc_count = 0
            for j in range(0, self.num_questions):
                id = self.current_proposals[j]
                score = self.input_data[i][j]
                ordered_score = np.argsort(score*(-1))
                #postion = ordered_score[index]
                #print(ordered_score.shape)
                #ordered_score = np.reshape(ordered_score,[-1,1])
                #print(ordered_score.shape)
                position = np.where(ordered_score == id)[0]
                #acc = self.acc_value[position]
                # standard_position = len(list(position) + 1) / 2
                # mean_postion = np.mean(position)
                # diff = mean_postion - standard_position
                # acc = np.median(1 / (0.1 * diff + 1))
                acc = longtailfunc(position)
                acc_count += acc
            ability = acc_count/self.num_questions
            abilities.append((ability, i))
        abilities = np.array(abilities)
        return abilities

    # def get_average_qdifficulty_3d(self):
    #     """
    #     Get average difficulty in 3D case
    #     """
    #     qdifficulties = []
    #     for i in range(0, self.num_questions):
    #         qdifficulty = 0
    #         for j in range(0, self.num_options):
    #             index_arr = np.intersect1d(np.where(
    #                 self.odifficulties[:, 1] == i), np.where(self.odifficulties[:, 2] == j))
    #             assert len(index_arr) == 1
    #             qdifficulty += self.odifficulties[index_arr[0]][0]
    #         qdifficulty = qdifficulty / float(self.num_options)
    #         qdifficulties.append((qdifficulty, i))
    #     qdifficulties = np.array(qdifficulties)
    #     return qdifficulties

    def get_average_odifficulty_3d(self):
        """
        Get average difficulty in 3D case
        """
        # acc_value = self.get_acc_value()
        odifficulties = []
        for i in range(0, self.num_questions):
            err_count = 0
            for k in range(0, self.num_participants):
                id = self.current_proposals[i]
                score = self.input_data[k][i]
                ordered_score = np.argsort(score*(-1))
                position = np.argwhere(ordered_score == id)[0]
                #acc = self.acc_value[position]
                # standard_position = len(list(position) + 1) / 2
                # mean_postion = np.mean(position)
                # diff = mean_postion - standard_position
                # acc = np.median(1 / (0.1 * diff + 1))
                acc = longtailfunc(position)
                err = 1 - acc

#                j = i%158##################################################################################
 
                err_count += err
            odifficulty = err_count/ self.num_participants
            #odifficulties.append((odifficulty, i, j))
            odifficulties.append((odifficulty, i))
        odifficulties = np.array(odifficulties)
        return odifficulties

    def get_buckets(self, attribute, k):
        """
        Get k buckets for the attribute list given in ascending order
        """
        attribute = attribute.tolist()
        attribute.sort(key=lambda x: x[0])
        attribute_step_size = len(attribute) // k
        for i in range(0, k):
            current_bucket = attribute[
                i * attribute_step_size:min((i + 1) * attribute_step_size, len(attribute))]
            for j in range(0, len(current_bucket)):
                current_bucket[j][1] = int(current_bucket[j][1])
                if len(current_bucket[j]) == 3:
                    current_bucket[j][2] = int(current_bucket[j][2])
            yield current_bucket

    def get_ability_in_bucket(self, odifficulty_bucket):
        """
        Get the ability in the bucket ability_bucket
        """
        # acc_value = self.get_acc_value()
        abilities = []
        for i in range(0, self.num_participants):
            acc_count = 0
            for j in range(0, len(odifficulty_bucket)):
                id = self.current_proposals[odifficulty_bucket[j][1]]
                score = self.input_data[i][odifficulty_bucket[j][1]]
                ordered_score = np.argsort(score*(-1))
                position = np.argwhere(ordered_score == id)[0]
                # standard_position = len(list(position) + 1) / 2
                # mean_postion = np.mean(position)
                # diff = mean_postion - standard_position
                # acc = np.median(1 / (0.1 * diff + 1))
                acc = longtailfunc(position)

                #acc = self.acc_value[position]
                acc_count += acc
            ability = acc_count / len(odifficulty_bucket)
            abilities.append((ability, i))
        abilities = np.array(abilities)
        return abilities


    def get_odifficulty_in_bucket(self, ability_bucket):
        """
        Get the difficulty in the bucket difficulty_bucket
        """
        # acc_value = self.get_acc_value()
        odifficulties = []
        for j in range(0, self.num_questions):
#            k = j%158#####################################################################################
            # k = (j%158)+158
            err_count = 0
            for i in range(0, len(ability_bucket)):
                id = self.current_proposals[j]
                score = self.input_data[ability_bucket[i][1]][j]
                ordered_score = np.argsort(score*(-1))
                position = np.argwhere(ordered_score == id)[0]
                # standard_position = len(list(position) + 1) / 2
                # mean_postion = np.mean(position)
                # diff = mean_postion - standard_position
                # acc = np.median(1 / (0.1 * diff + 1))
                acc = longtailfunc(position)
                #acc = self.acc_value[position]
                err = 1 - acc
                err_count += err
            odifficulty = err_count / len(ability_bucket)
            #odifficulties.append((odifficulty, j, k))
            odifficulties.append((odifficulty, j))
        odifficulties = np.array(odifficulties)
        return odifficulties


    def generate_features_3d(self, input_data, current_proposals, acc_value):
        """
        Generate the feature input given the data and current proposed
        answer sheet
        """
        start_time1 = time.clock()
        assert len(np.shape(input_data)) == self.num_dimensions
        assert len(current_proposals) == self.num_questions
        #assert len(current_proposals[0]) == self.num_options
        self.input_data = input_data
        self.current_proposals = current_proposals
        self.acc_value = acc_value
        self.abilities = self.get_average_ability_3d()
        self.odifficulties = self.get_average_odifficulty_3d()
        #self.qdifficulties = self.get_average_qdifficulty_3d()
        abilities_bucket_wise = []
        qdifficulties_bucket_wise = []
        odifficulties_bucket_wise = []
        for ability_bucket in self.get_buckets(self.abilities[:], self.k_ability):
            odifficulties_bucket_wise.append(
                self.get_odifficulty_in_bucket(ability_bucket))
        for odifficulty_bucket in self.get_buckets(self.odifficulties[:], self.k_odifficulty):
            abilities_bucket_wise.append(
                self.get_ability_in_bucket(odifficulty_bucket))
        self.abilities_bucket_wise = np.array(abilities_bucket_wise)
        self.odifficulties_bucket_wise = np.array(odifficulties_bucket_wise)
        end_time1 = time.clock()
        print(end_time1 - start_time1)

    def get_features_3d_element(self, person, question):
        """
        Get a particular value in the generated feature matrix
        """
        feature_list = []

        feature_list.append(self.abilities[person][0])
        for i in range(0, self.k_odifficulty):
            feature_list.append(self.abilities_bucket_wise[i][person][0])

        feature_list.append(self.odifficulties[question][0])
        for i in range(0, self.k_ability):
            feature_list.append(self.odifficulties_bucket_wise[i][question][0])
        feature_list = np.array(feature_list)
        return feature_list

    def get_features_3d(self, input_data, galley_labels, probe_labels, acc_value):
        """
        Get the feature list by flattening the feature matrix
        """
        features_list = []
        labels_list = []
        for i in range(0, self.num_participants):
            start_time2 = time.clock()
            print('i:',i)
            for j in range(0, self.num_questions):
                #k = j%158####################################################################################
                # k = (j%158)+158
                probe_label = probe_labels[j]

                features_list.append(self.get_features_3d_element(i, j))
                score = input_data[i][j]
                ordered_score = np.argsort(score * (-1))
                gallery_label = []
                for k in range(0, self.num_options):
                    idex = ordered_score[k]
                    g_label = galley_labels[idex]
                    gallery_label.append(g_label)				
                position = np.where(gallery_label == probe_label)[0]
                standard_position = float(len(position)-1)/2
                mean_postion = np.mean(position)
                diff = mean_postion - standard_position
                label = np.median(longtailfunc(diff))
                #label = self.acc_value[position]
                labels_list.append(label)
                if j%100 == 0:
                    print('>>>worker:'+str(i)+'/88'+'>>>query:'+str(j)+'/1684')
            end_time2 = time.clock()
            print(end_time2-start_time2)

        features_list = np.array(features_list)
        labels_list = np.array(labels_list)
        print('features_list:',features_list.shape)
        return features_list, labels_list

