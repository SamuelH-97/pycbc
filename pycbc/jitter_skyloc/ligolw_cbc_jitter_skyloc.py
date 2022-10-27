#!/usr/bin/env python
# Copyright (C) 2010  Nickolas Fotopoulos
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
Given an XML file with a SimInspiralTable, add a random offset to each
injection's sky locations.
"""
from __future__ import division

__author__ = "Nickolas Fotopoulos <nvf@gravity.phys.uwm.edu>"

import math
import argparse
import random
import sys

import numpy as np

import lal
from glue.ligolw import ligolw
from glue.ligolw import lsctables
from glue.ligolw import table
from glue.ligolw import utils
from glue.ligolw.utils import process as ligolw_process
#removed
#from pylal import git_version
#from pylal import sphericalutils as su
#from pylal import inject
#added
import pycbc.version as ver
from pycbc.detector import Detector as py_det
from lal import cached_detector_by_name
#
from lal import LIGOTimeGPS
lsctables.LIGOTimeGPS = LIGOTimeGPS
eff_distance = py_det.efective_distance


#
# Utility functions
#
def polaz2lonlat(theta, phi):
    """
    Convert (polar, azimuthal) angles in radians to (longitude, latitude)
    in radians.
    """
    return phi, lal.PI_2 - theta

def lonlat2polaz(lon, lat):
    """
    Convert (longitude, latitude) in radians to (polar, azimuthal) angles
    in radians.
    """
    return lal.PI_2 - lat, lon

def site_end_time(detector, sim):
    """
    Return the time of the simulated gravitational wave arriving
    the given detector, computed from the arrival time at geocenter.
    """
    delay = LIGOTimeGPS(lal.TimeDelayFromEarthCenter(detector.location,
        sim.longitude, sim.latitude, sim.get_end()))
    return sim.get_end() + delay

#def eff_distance(detector, sim):
#    """
#    Return the effective distance.
#
#    Ref: Duncan's PhD thesis, eq. (4.3) on page 57, implemented in
#         LALInspiralSiteTimeAndDist in SimInspiralUtils.c:594.
#    """
#    #....changing inject.XLALComputeDetAMResponse to lal.ComputeDetAMResponse
#    f_plus, f_cross = lal.ComputeDetAMResponse(detector.response,
#        sim.longitude, sim.latitude, sim.polarization, sim.end_time_gmst)
#    ci = math.cos(sim.inclination)
#    s_plus = -(1 + ci * ci)
#    s_cross = -2 * ci
#    return 2 * sim.distance / math.sqrt(f_plus * f_plus * s_plus * s_plus + \
#        f_cross * f_cross * s_cross * s_cross)
#....adding definitions fisher_rvs, new_z_to_euler, rotate_euler
def fisher_rvs(mu, kappa, size=1):
    """
    Return a random (polar, azimuthal) angle drawn from the Fisher
    distribution. Assume that the concentration parameter (kappa) is large
    so that we can use a Rayleigh distribution about the north pole and
    rotate it to be centered at the (polar, azimuthal) coordinate mu.

    Assume kappa = 1 / sigma**2

    References:
      * http://en.wikipedia.org/wiki/Von_Mises-Fisher_distribution
      * http://arxiv.org/pdf/0902.0737v1 (states the Rayleigh limit)
    """
    rayleigh_rv = \
        np.array((np.random.rayleigh(scale=1. / np.sqrt(kappa), size=size),
                  np.random.uniform(low=0, high=2*lal.PI, size=size)))\
                .reshape((2, size)).T  # guarantee 2D and transpose
    a, b = new_z_to_euler(mu)
    return rotate_euler(rayleigh_rv, a, b, 0)
def new_z_to_euler(new_z):
    """
    From the new Z axis expressed in (polar, azimuthal) angles of the
    initial coordinate system, return the (alpha, beta) Euler angles
    that rotate the old Z axis (0, 0) to the new Z axis.
    """
    return (lal.PI_2 + new_z[1]) % (2 * lal.PI), new_z[0]
def rotate_euler(sph_coords, alpha, beta, gamma):
    """
    Take an Nx2 array of N (theta, phi) vectors on the unit sphere
    (that is, (polar, azimuthal) angles in radians) and apply
    rotations through the Euler angles alpha, beta, and gamma
    in radians, using the ZXZ convention.
    """
    c = np.cos
    s = np.sin

    # Define rotation matrix
    R = np.array(
        [[c(alpha) * c(gamma) - c(beta) * s(alpha) * s(gamma),
          c(gamma) * s(alpha) + c(alpha) * c(beta) * s(gamma),
          s(beta) * s(gamma)],
         [-c(beta) * c(gamma) * s(alpha) - c(alpha) * s(gamma),
          c(alpha) * c(beta) * c(gamma) - s(alpha) * s(gamma),
          c(gamma) * s(beta)],
         [s(alpha) * s(beta), -c(alpha) * s(beta), c(beta)]], dtype=float)

    # Convert to intermediate cartesian representation (N x 3)
    cart_orig = np.empty(shape=(len(sph_coords), 3), dtype=float)
    cart_orig[:, 0] = c(sph_coords[:, 1]) # * s(sph_coords[:, 0])
    cart_orig[:, 1] = s(sph_coords[:, 1]) # * s(sph_coords[:, 0])
    cart_orig[:, 0:2] *= s(sph_coords[:, 0])[:, None]
    cart_orig[:, 2] = c(sph_coords[:, 0])

    # Rotate by x_i = R_{ij} * x_j
    # NB: dot contracts last dim of A with second-to-last dim of B
    cart_new = np.dot(cart_orig, R)

    # Extract new spherical coordinates
    sph_new = np.empty(shape=(len(sph_coords), 2), dtype=float)
    sph_new[:, 0] = np.arccos(cart_new[:, 2])
    sph_new[:, 1] = np.arctan2(cart_new[:, 1], cart_new[:, 0])

    return sph_new
#
# Parse commandline
#

parser = argparse.ArgumentParser()
#....changing git_version
parser.add_argument("-v", "--version", action="version",
                    version=ver.git_verbose_msg)
parser.add_argument("--input-file", metavar="INFILE", required=True,
                    help="path of input XML file")
parser.add_argument("--output-file", metavar="OUTFILE", required=True,
                    help="path of output XML file")

s_group = parser.add_argument_group("Sky position arguments")
s_group.add_argument("--jitter-sigma-deg", metavar="JSIG", type=float,
                     default=0,
                     help="jitter injections with a Gaussian of standard "
                          "deviation JSIG degrees")
s_group.add_argument("--apply-fermi-error", action="store_true",
                     help="applies the statistical error from Fermi according "
                          "to https://wiki.ligo.org/foswiki/bin/view/Bursts/"
                          "S6VSR2InjectionSkyPositionJitter")
s_group.add_argument("--s6-systematics", action="store_true",
                     help="Applies the systematic Fermi error used in S6")

d_group = parser.add_argument_group("Distance arguments")
d_group.add_argument("--d-distr", default="gaussian",
                     choices=["gaussian", "lognormal"],
                     help="The distribution from which to draw the random "
                          "jittered ditance")
d_group.add_argument("--phase-error", type=float, default=0,
                     help="Phase calibration uncertainty (degrees)")
d_group.add_argument("--amplitude-error", type=float, default=0,
                     help="Amplitude calibration uncertainty (percent)")

args = parser.parse_args()

siminsp_fname = args.input_file
jitter_sigma = np.radians(args.jitter_sigma_deg)

# for details on those parameters see 
# https://wiki.ligo.org/foswiki/bin/view/Bursts/S6VSR2InjectionSkyPositionJitter
if args.s6_systematics:
    sys_core = np.radians(3.2)
    sys_tail = np.radians(9.5)
    tail_frac = 0.3
else:
    sys_core = np.radians(3.7)
    sys_tail = np.radians(14.0)
    tail_frac = 0.1

jitter_fermi_core = np.sqrt(jitter_sigma**2 + sys_core**2)
jitter_fermi_tail = np.sqrt(jitter_sigma**2 + sys_tail**2)

#
# Read inputs
#

siminsp_doc = utils.load_filename(siminsp_fname,
    gz=siminsp_fname.endswith(".gz"), contenthandler = lsctables.use_in(ligolw.LIGOLWContentHandler))

# Prepare process table with information about the current program
#....changing version, cvs_entry_time
process = ligolw_process.register_to_xmldoc(siminsp_doc,
    "ligolw_cbc_jitter_skyloc",
    args.__dict__, version=ver.git_tag,
    cvs_repository="lalsuite", cvs_entry_time=ver.date)


#
# Compute new sky locations
#
# ....removing inject ...
cached_detector = cached_detector_by_name
site_location_list = [\
    ("h", cached_detector["LHO_4k"]),
    ("l", cached_detector["LLO_4k"]),
    ("g", cached_detector["GEO_600"]),
    ("t", cached_detector["TAMA_300"]),
    ("v", cached_detector["VIRGO"])]
for sim in table.get_table(siminsp_doc, lsctables.SimInspiralTable.tableName):

    # The Fisher distribution is the appropriate generalization of a
    # Gaussian on a sphere.
    if args.apply_fermi_error:
      jitter_sig = np.where(random.random() < tail_frac, jitter_fermi_tail,
                            jitter_fermi_core)
    else:
      jitter_sig = jitter_sigma

    if jitter_sig > 0:
      kappa = 1 / (0.66 * jitter_sig)**2  # approximation from Briggs99
      #....modified.........
      kappa = 1 / (0.66 * jitter_sig)**2  # approximation from Briggs99
      fisher=fisher_rvs(np.array(lonlat2polaz(sim.longitude, sim.latitude)), kappa,size=1).squeeze()
      fisher_theta=fisher[0]
      fisher_phi=fisher[1]
      sim.longitude, sim.latitude = polaz2lonlat(fisher_theta,fisher_phi)
    
      # update arrival times and effective distances at the sites
      sim_geocent_end = sim.get_end()
      for site, detector in site_location_list:
          site_end = site_end_time(detector, sim)
          setattr(sim, site + "_end_time", site_end.gpsSeconds)
          setattr(sim, site + "_end_time_ns", site_end.gpsNanoSeconds)
          setattr(sim, "eff_dist_" + site, eff_distance(detector, sim.distance, sim.longitude, sim.latitude, sim.polarization, sim.end_time_gmst, sim.inclination))

#
# Compute new injected distances
#
if args.phase_error != 0 or args.amplitude_error != 0:
  for sim in table.get_table(siminsp_doc,
                             lsctables.SimInspiralTable.tableName):
    mu = sim.distance * (1 + 0.5 * np.deg2rad(args.phase_error)**2)
    sigma = (args.amplitude_error / 100.) * sim.distance
    if args.d_distr == "gaussian":
      setattr(sim, "distance", random.gauss(mu, sigma))
      for site, detector in site_location_list:
        setattr(sim, "eff_dist_" + site, eff_distance(detector, sim))
    elif args.d_distr == "lognormal":
      setattr(sim, "distance", np.log(random.lognormvariate(mu, sigma)))
      for site, detector in site_location_list:
        setattr(sim, "eff_dist_" + site, eff_distance(detector, sim))

#
# Write output
#

ligolw_process.set_process_end_time(process)
utils.write_filename(siminsp_doc, args.output_file)
