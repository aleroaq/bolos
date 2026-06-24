from collections import defaultdict
import logging

import numpy as np

from bolos.process import Process


class Target(object):
    """ A class to contain all information related to one target. """

    def __init__(self, name: str):
        """ Initializes an instance of target named name."""
        self.name = name
        self.mass_ratio = None
        self.density = 0.0

        # Lists of all processes pertaining this target
        self.elastic : list[Process]             = []
        self.effective : list[Process]           = []
        self.attachment : list[Process]          = []
        self.ionization : list[Process]          = []
        self.excitation : list[Process]          = []
        self.weighted_elastic : list[Process]    = []
        self.coulomb : list[Process]             = []

        self.kind = {
            'ELASTIC': self.elastic,
            'EFFECTIVE': self.effective,
            'MOMENTUM': self.effective,
            'ATTACHMENT': self.attachment,
            'IONIZATION': self.ionization,
            'EXCITATION': self.excitation,
            'WEIGHTED_ELASTIC': self.weighted_elastic,
            'COULOMB': self.coulomb,
        }

        self.by_product = defaultdict(list)

        logging.debug("Target %s created." % str(self))

    def add_process(
        self, 
        process: Process,
    ):
        kind = self.kind[process.kind]
        kind.append(process)

        if process.mass_ratio is not None:
            logging.debug("Mass ratio (=%g) for %s" 
                          % (process.mass_ratio, str(self)))

            if (self.mass_ratio is not None 
                and self.mass_ratio != process.mass_ratio):
                raise ValueError("More than one mass ratio for target '%s'"
                                 % self.name)

            self.mass_ratio = process.mass_ratio

        process.target = self

        self.by_product[process.product].append(process)

        logging.debug("Process %s added to target %s" 
                      % (str(process), str(self)))

    def ensure_elastic(self) -> None:
        """ Makes sure that the process has an elastic cross-section.
        If the user has specified an effective cross-section, we remove
        all the other cross-sections from it. """
        if self.elastic and self.effective:
            raise ValueError("In target '%s': EFFECTIVE/MOMENTUM and ELASTIC "
                             "cross-sections are incompatible." % self)
        
        if self.elastic and self.coulomb:
            raise ValueError("In target '%s': COULOMB and ELASTIC "
                             "cross-sections are incompatible." % self)
        
        if self.effective and self.coulomb:
            raise ValueError("In target '%s': COULOMB and EFFECTIVE "
                             "cross-sections are incompatible." % self)

        if self.elastic or self.coulomb:
            return

        if len(self.effective) > 1:
            raise ValueError("In target '%s': Can't handle more that 1 "
                             "EFFECTIVE/MOMENTUM for a given target" % self)
            
        if not self.effective and not self.coulomb:
            logging.warning("Target %s has no ELASTIC or EFFECTIVE or COULOMB "
                            "cross sections" % str(self))
            return

        newdata = self.effective[0].data.copy()
        for p in self.inelastic:
            newdata[:, 1] -= p.interp(newdata[:, 0])

        if np.amin(newdata[:, 1]) < 0:
            # logging.warning('After substracting INELASTIC from EFFECTIVE, '
            #                 'target %s has negative cross-section.'
            #                 % self.name)
            # logging.warning('Setting as max(0, ...)')
            newdata[:, 1] = np.where(newdata[:, 1] > 0, newdata[:, 1], 0)


        newelastic = Process(
            target=self.name, 
            kind='ELASTIC',
            data=newdata,
            mass_ratio=self.effective[0].mass_ratio,
            comment="Calculated from EFFECTIVE cross sections")

        logging.debug("EFFECTIVE -> ELASTIC for target %s" % str(self))
        self.add_process(newelastic)

        # Remove the EFFECTIVE processes.
        self.effective: list[Process] = []


    @property
    def inelastic(self) -> list[Process]: 
        """ An useful abbreviation. """
        return (self.attachment + self.ionization + self.excitation)

    @property
    def everything(self) -> list[Process]:
        """ A list with ALL processes.  We do not use all as a name
        to avoid confusion with the python function."""
        return (
            self.elastic + self.attachment + 
            self.ionization + self.excitation + 
            self.coulomb
        )

    def __repr__(self) -> str:
        return "Target(%s)" % repr(self.name)

    def __str__(self) -> str:
        return self.name

