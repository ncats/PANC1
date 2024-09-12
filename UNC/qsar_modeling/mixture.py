import numpy as np
import pandas as pd
import math
import sys
from mol import Mol

class Mixture:

    def __init__(self, mol_a, mol_b, score = None, identifier = None):

        self.mol_a = mol_a
        self.mol_b = mol_b

        if not identifier:
            if self.mol_a.identifier == None or self.mol_b.identifier == None:
                raise Exception("No identifier provided and at least one mol has no identifier")

            identifier = self.mol_a.identifier + "|" + self.mol_b.identifier

        self.identifier = identifier

        self.score = score

    #descriptor_function should accept two Mol objects
    def get_descriptor_with_handler(self, descriptor_handler, descriptor_function):

        return descriptor_handler(self.mol_a, self.mol_b, descriptor_function)

    def get_descriptor(self, descriptor_function):

        return descriptor_function(self.mol_a, self.mol_b)

    def copy(self):

        m = Mixture(self.mol_a, self.mol_b, score = self.score)
        return m

    def has_molecule_id(self, identifier):

        print(self.mol_a.identifier, self.mol_b.identifier, identifier)
        if self.mol_a.identifier == identifier or \
            self.mol_b.identifier == identifier:
                return True
        return False

    def __repr__(self):
        return f"{self.mol_a}|{self.mol_b}"

    def __str__(self):
        return self.__repr__()


class EugeneSplit:

    def __init__(self, mixtures):
        a_ids = set([m.mol_a.identifier for m in mixtures])
        b_ids = set([m.mol_b.identifier for m in mixtures])
        all_ids = a_ids.union(b_ids)
        self.ids = all_ids
        self.mixtures = [mixture.copy() for mixture in mixtures]


    #assumes relatively dense matrix
    #randomly partition all ids into two sets, and return all mixtures contained only within each set
    def sample(self):

        ids = set(self.ids)
        np_ids = np.array(list(ids))

        num_to_sample = int(np.rint(len(np_ids) / 2))
        partition_a = set(np.random.choice(np_ids, size = num_to_sample, replace = False))
        partition_b = ids - partition_a

        set_a = []
        set_b = []
        compounds_out = []
        for mixture in self.mixtures:
            a_count = sum([mixture.mol_a.identifier in partition_a, mixture.mol_b.identifier in partition_a])
            b_count = sum([mixture.mol_a.identifier in partition_b, mixture.mol_b.identifier in partition_b])
            assert(a_count + b_count == 2) #no individual compound should be in partition a and b

            if  a_count == 2:
                set_a.append(mixture)
            elif b_count == 2:
                set_b.append(mixture)
            elif a_count == 1 and b_count == 1:
                compounds_out.append(mixture)
            else:
                raise Exception("Mixture not fully in partitions a or b, or shared between them")

        return set_a, set_b, compounds_out


#not really leave "one" out, becuase a single compound is chosen, and all mixtures
#containing it are sent to the test set
#everything else goes to the train set
class CompoundLeaveOneOut:

    def __init__(self, mixtures):
        a_ids = set([m.mol_a.identifier for m in mixtures])
        b_ids = set([m.mol_b.identifier for m in mixtures])
        all_ids = a_ids.union(b_ids)
        self.ids = list(all_ids)
        self.mixtures = [mixture.copy() for mixture in mixtures]
        self.iteration_counter = 0


    def __iter__(self):
        return self

    def __next__(self):

        if self.iteration_counter == len(self.ids):
            self.iteration_counter = 0
            raise StopIteration

        held_out_id = self.ids[self.iteration_counter]

        train = set()
        test = set()

        for mixture in self.mixtures:
            if mixture.mol_a.identifier == held_out_id or mixture.mol_b.identifier == held_out_id:
                test.add(mixture)
            else:
                train.add(mixture)

        self.iteration_counter += 1

        return train, test

class PairLeaveOneOut:

    def __init__(self, mixtures):
        a_ids = set([m.mol_a.identifier for m in mixtures])
        b_ids = set([m.mol_b.identifier for m in mixtures])
        all_ids = a_ids.union(b_ids)
        self.ids = list(all_ids)
        self.mixtures = [mixture.copy() for mixture in mixtures]
        self.iteration_counter = 0


    def __iter__(self):
        return self

    #could probably just pick one mixture and train on all others, but checking all ids
    #will handle the case where multiple concentrations of a mixture are present
    def __next__(self):

        if self.iteration_counter == len(self.mixtures):
            self.iteration_counter = 0
            raise StopIteration

        held_out_mixture = self.mixtures[self.iteration_counter]
        held_out_compound_a = held_out_mixture.mol_a.identifier
        held_out_compound_b = held_out_mixture.mol_b.identifier
        held_out_pair = set([held_out_compound_a, held_out_compound_b])

        train = set()
        test = set()
        ignored = set()

        for mixture in self.mixtures:
            if mixture.mol_a.identifier in held_out_pair and mixture.mol_b.identifier in held_out_pair:
                test.add(mixture)
            elif mixture.mol_a.identifier in held_out_pair or mixture.mol_b.identifier in held_out_pair:
                ignored.add(mixture)
            else:
                train.add(mixture)

        self.iteration_counter += 1

        return train, test, ignored


def get_mixture_descriptors(mixtures, descriptor_function, handler = None):

    if handler:
        descriptors = [m.get_descriptor_with_handler(handler, descriptor_function) for m in mixtures]
    else:
        descriptors = [m.get_descriptor(descriptor_function) for m in mixtures]

    descriptors = np.array(descriptors)
    scores = [m.score for m in mixtures]

    return descriptors, scores

def concatenate_descriptors(mol_a, mol_b, descriptor_function):

    desc_a = descriptor_function(mol_a)
    desc_b = descriptor_function(mol_b)

    return np.concatenate((desc_a, desc_b))

def concatenate_descriptors_backwards(mol_a, mol_b, descriptor_function):

    desc_a = descriptor_function(mol_a)
    desc_b = descriptor_function(mol_b)

    return np.concatenate((desc_b, desc_a))

def sum_descriptors(mol_a, mol_b, descriptor_function):

    desc_a = descriptor_function(mol_a)
    desc_b = descriptor_function(mol_b)

    return np.add(desc_b, desc_a)

def average_descriptors(mol_a, mol_b, descriptor_function):

    desc_a = descriptor_function(mol_a)
    desc_b = descriptor_function(mol_b)

    a = np.vstack((desc_a, desc_b))

    return np.mean(a, axis = 0)

def concatenate_descriptors_with_shared(mol_a, mol_b, descriptor_function):

    desc_a = descriptor_function(mol_a)
    desc_b = descriptor_function(mol_b)
    shared = np.logical_and(desc_a, desc_b)

    return np.concatenate((desc_a, desc_b, shared))

def concatenate_descriptors_backwards_with_shared(mol_a, mol_b, descriptor_function):

    desc_a = descriptor_function(mol_a)
    desc_b = descriptor_function(mol_b)
    shared = np.logical_and(desc_a, desc_b)

    return np.concatenate((desc_b, desc_a, shared))




