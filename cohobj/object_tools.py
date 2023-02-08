"""
object_tools module.

@author: Peter Clark
"""
import numpy as np
import xarray as xr
from scipy import ndimage

def label_3D_cyclic(mask) :
    """
    Label 3D objects taking account of cyclic boundary in x and y.

    Uses ndimage(label) as primary engine.

    Parameters
    ----------
        mask: xarray.DataArray bool
            3D logical array with object mask (i.e. objects are
            contiguous True).

    Returns
    -------
        Object identifiers::

            labs  : Integer array[nx,ny,nz] of labels. -1 denotes unlabelled.
            nobjs : number of distinct objects. Labels range from 0 to nobjs-1.

    @author: Peter Clark

    """
    (nx, ny, nz) = mask.shape
    labels, nobjects = ndimage.label(mask)
    labels -=1
    
    def relabel(labs, nobjs, i,j) :
        lj = (labs == j)
        labs[lj] = i
        for k in range(j+1,nobjs) :
            lk = (labs == k)
            labs[lk] = k-1
        nobjs -= 1
        return labs, nobjs

    def find_objects_at_edge(minflag, dim, n, labs, nobjs) :
        #debug_label = True
        debug_label = False

        i = 0
        while i < (nobjs-2) :
            # grid points corresponding to label i
            posi = np.where(labs == i)
            posid = posi[dim]
            # Does object i have any points on the required border?
            if minflag :
                obj_i_on_border = (np.min(posid) == 0)
                border = '0'
            else:
                obj_i_on_border = (np.max(posid) == (n-1))
                border = f"n{['x','y'][dim]}-1"
                
            if obj_i_on_border :
                if debug_label :
                    print(f"Object {i:03d} on {['x','y'][dim]}={border} border?")
                        
                # If object i does have any points on the required border
                # Loop over remaining objects to see if they have points
                # on the opposite border
                j = i+1
                while j < nobjs :
                    # grid points corresponding to label j
                    posj = np.where(labs == j)
                    posjd = posj[dim]

                    if minflag :
                        obj_j_on_opposite_border = (np.max(posjd) == (n-1))
                        border = f"n{['x','y'][dim]}-1"
                    else:
                        obj_j_on_opposite_border = (np.min(posjd) == 0)
                        border = '0'

                    if obj_j_on_opposite_border :
                        # If object i does have any points on the 
                        # opposite border, then do they overlap in the
                        # other horizontal coordinate space?

                        if debug_label :
                            print(f"Match Object {j:03d} on {['x','y'][dim]}={border} border?")

                        # Select out just the border points for object i and j.
                        if minflag :
                            ilist = np.where(posid == 0)
                            jlist = np.where(posjd == (n-1))
                        else :
                            ilist = np.where(posid == (n-1))
                            jlist = np.where(posjd == 0)
                                                        
                        # x or y intersection
                        int1 = np.intersect1d(posi[1-dim][ilist],
                                              posj[1-dim][jlist])
                        # z-intersection
                        int2 = np.intersect1d(posi[2][ilist],
                                              posj[2][jlist])
                        # If any overlab, label object j as i and
                        # relabel the rest.
                        if np.size(int1)>0 and np.size(int2)>0 :
                            if debug_label :
                                print('Yes!',i,j)
                            labs, nobjs = relabel(labs, nobjs, i, j)
                    j += 1
            i += 1
        return labs, nobjs
    # Look at 4 boundary zones, i in [0, nx), j in [0, ny).
    labels, nobjects = find_objects_at_edge(True,  0, nx, labels, nobjects)
    labels, nobjects = find_objects_at_edge(False, 0, nx, labels, nobjects)
    labels, nobjects = find_objects_at_edge(True,  1, ny, labels, nobjects)
    labels, nobjects = find_objects_at_edge(False, 1, ny, labels, nobjects)
    
    labels = xr.DataArray(labels, 
                          name='object_labels', 
                          coords=mask.coords, 
                          dims=mask.dims,
                          attrs={'nobjects':nobjects},
                          )

    return labels

def get_object_labels(mask: xr.DataArray)->xr.DataArray:
    """
    Convert 3D logical mask to object labels corresponding to mask positions. 

    Parameters
    ----------
    mask : xr.DataArray
        Evaluates True at required positions.

    Returns
    -------
    object_labels : xr.DataArray (int32)
        Contains data variables "object_labels", counting from 0.
        Coordinates "pos_number" and any others (e.g. "time") in mask.

    """        
    olab = label_3D_cyclic(mask)
    
    
    olab = ( olab.where(mask, drop=True)
                 .stack(pos_number=("x", "y", "z"))
                 .dropna(dim="pos_number")
           )

    olab = ( olab.assign_coords(pos_number=np.arange(olab.pos_number.size))
                 .astype(int)
           )
            
    return olab

def unsplit_objects(ds_traj, Lx=None, Ly=None) :
    """
    Unsplit a set of objects at a set of times using unsplit_object on each.

    Parameters
    ----------
        ds_traj     : xarray dataset
            Trajectory points "x", "y", and "z" and "object_label".
        Lx,Ly   : Domain size in x and y directions.

    Returns
    -------
        Trajectory array with modified positions.

    @author: Peter Clark

    """
    nobjects = ds_traj["object_label"].attrs["nobjects"]
    if nobjects < 1:
        return ds_traj

    if Lx is None : Lx = ds_traj.attrs["Lx"]
    if Ly is None : Ly = ds_traj.attrs["Ly"]

    print('Unsplitting Objects:')
    
    def _unsplit_object(tr):
        """
        Gather together points in object separated by cyclic boundaries.

            For example, if an object spans the 0/Lx boundary, so some
            points are close to zero, some close to Lx, they will be adjusted to
            either go from negative to positive, close to 0, or less than Lx to
            greater than. The algorithm is very simple. First, we only consider
            sets of points whose span is greater than L/2, then either shift  
            in points in [0,L/4) to [L,5L/4) or from [3L/4,L) to [-L/4,0), 
            depending on which set of points has smaller number.

        Parameters
        ----------
            tr      : xr.Dataset
                Trajectory points belonging to object.

        Returns
        -------
            tr      : xr.Dataset
                Adjusted trajectory points in object.

        @author: Peter Clark

        """

        L = (Lx, Ly)
        for it in range(0, ds_traj["time"].size) :
            for idim, dim in enumerate('xy'):
                v = tr[dim].isel(time=it)
                if v.max() - v.min() > L[idim]/2: 
                    q0 = v < L[idim] * 0.25
                    q3 = v > L[idim] * 0.75
                    if q0.sum() < q3.sum():
                        tr[dim].isel(time=it)[q0] += L[idim]
                    else:
                        tr[dim].isel(time=it)[q3] -= L[idim]
                    
        return tr
    
    ds_traj = ds_traj.groupby("object_label").map(_unsplit_object)
        
    return ds_traj

def get_bounding_boxes(ds_traj, use_mask=False):
    """
    Find x,y,z min and max for objects in ds_traj.

    Parameters
    ----------
    ds_traj : xarray.Dataset
        Trajectory points "x", "y", "z" 
        with additional non-dim coord "object_label" to groupby objects.
        and optional "object_mask" boolean data_var.
    use_mask : bool, optional 
        If true, only use points masked True by "object_mask". 
        The default is False.

    Returns
    -------
    ds_out : xarray.Dataset
        Contains {x/y/z}_{min/max/mean}

    """
    
    if use_mask:
        ds = ds_traj[["x", "y", "z", "object_label"]].where(ds_traj["object_mask"]).groupby("object_label")
    else:    
        ds = ds_traj[["x", "y", "z", "object_label"]].groupby("object_label")

    ds_min = ds.min(dim="trajectory_number")
    ds_max = ds.max(dim="trajectory_number")
    ds_mean = ds.mean(dim="trajectory_number")
    
    for c in 'xyz':
        ds_min = ds_min.rename({c:f"{c}_min"})
        ds_max = ds_max.rename({c:f"{c}_max"})
        ds_mean = ds_mean.rename({c:f"{c}_mean"})
        
    ds_out = xr.merge([ds_min, ds_max, ds_mean])
    ds_out.attrs = ds_traj.attrs

    return ds_out

def box_bounds(b:xr.Dataset)->xr.Dataset:
    """
    Select object box boundaries from Dataset.

    Parameters
    ----------
    b : xr.Dataset
        Dataset containing box boundaries.

    Returns
    -------
    box : xr.Dataset
        Dataset containing just box boundaries.

    """

    box = b[['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max']]

    return box
            
def box_xyz(b):
    """
    Convert object bounds to plottable x,y,z for a box.

    Parameters
    ----------
    b : xarray.Dataset
        Contains {x/y/z}_{min/max}

    Returns
    -------
    x : numpy array
        x values for box.
    y : numpy array
        y values for box.
    z : numpy array
        z values for box.

    """
    x = np.array([b.x_min, b.x_min, b.x_max, b.x_max, b.x_min,
                  b.x_min, b.x_min, b.x_max, b.x_max, b.x_min])
    y = np.array([b.y_min, b.y_max, b.y_max, b.y_min, b.y_min,
                  b.y_min, b.y_max, b.y_max, b.y_min, b.y_min])
    z = np.array([b.z_min, b.z_min, b.z_min, b.z_min, b.z_min,
                  b.z_max, b.z_max, b.z_max, b.z_max, b.z_max])
    return x, y, z

def box_overlap_fast(test_xmin, test_xmax, test_ymin, test_ymax,
                     set_xmin,  set_xmax,  set_ymin,  set_ymax, set_id):

    def over_1D(min1, max1, min2, max2):

        # min1_ge_min2 =
        t1 = np.logical_and(min2 <= min1, min1 <= max2)
        t2 = np.logical_and(min2 <= max1, max1 <= max2)
        t3 = np.logical_and(min1 <= min2, min2 <= max1)
        t4 = np.logical_and(min1 <= max2, max2 <= max1)
        # t3 = np.logical_and(min1 <= min2, max1 >= max2)
        # t4 = np.logical_and(min1 >= min2, max1 <= max2)
        overlap =np.logical_or(np.logical_or( t1, t2), np.logical_or( t3, t4) )
        return overlap

    x_overlap = over_1D(test_xmin, test_xmax,
                        set_xmin,  set_xmax)

    y_overlap = over_1D(test_ymin, test_ymax,
                        set_ymin,  set_ymax)

    overlap = np.logical_and(x_overlap, y_overlap)

    overlap_boxes = set_id[overlap]

    return overlap_boxes

def box_overlap_with_wrap(b_test, b_set, nx, ny) :
    """
    Compute whether rectangular boxes intersect.

    Parameters
    ----------
        b_test: box for testing xarray.DataArray  
        b_set: set of boxes xarray.DataArray
        nx: number of points in x grid.
        ny: number of points in y grid.

    Returns
    -------
        overlapping box ids

    @author: Peter Clark

    """
    # Wrap not yet implemented

    def overlap_1D(min1, max1, min2, max2, n):

        # min1_ge_min2 =
        t1 = np.logical_and(min2 <= min1, min1 <= max2)
        t2 = np.logical_and(min2 <= max1, max1 <= max2)
        t3 = np.logical_and(min1 <= min2, min2 <= max1)
        t4 = np.logical_and(min1 <= max2, max2 <= max1)
        # t3 = np.logical_and(min1 <= min2, max1 >= max2)
        # t4 = np.logical_and(min1 >= min2, max1 <= max2)
        overlap =np.logical_or(np.logical_or( t1, t2), np.logical_or( t3, t4) )
        return overlap

    x_overlap = overlap_1D(b_test.x_min, b_test.x_max,
                            b_set.x_min,  b_set.x_max, nx)

    y_overlap = overlap_1D(b_test.y_min, b_test.y_max,
                            b_set.y_min,  b_set.y_max, ny)

    overlap = np.logical_and(x_overlap, y_overlap)

    overlap_boxes = b_set.where(overlap).dropna(dim="object_label")

    if overlap_boxes.object_label.size == 0: overlap_boxes = None

    return overlap_boxes

def refine_object_overlap_fast(tr1, tr2, nx, ny) :
    """
    Estimate degree of overlap between two trajectory objects.   

    Parameters
    ----------
    tr1 : np.array
        Trajectory data. [0:3,...] = [x, y, z] in grid points.
    tr2 : np.array
        Trajectory data. [0:3,...] = [x, y, z] in grid points.


    Returns
    -------
    float
        Fractional overlap.

    """
    def extract_obj_as1Dint(traj) :

        mask = traj['mask']
        tr = traj['xyz']

        itrx = (tr[0, mask] + 0.5).astype(int)
        itry = (tr[1, mask] + 0.5).astype(int)
        itrz = (tr[2, mask] + 0.5).astype(int)

        tr1D = np.unique(itrx + nx * (itry + ny * itrz))
        return tr1D

    tr1D  = extract_obj_as1Dint(tr1)
    trm1D = extract_obj_as1Dint(tr2)

    max_size = np.max([np.size(tr1D),np.size(trm1D)])
    if max_size > 0 :
        intersection = np.size(np.intersect1d(tr1D, trm1D)) / max_size
    else :
        intersection = 0
    return intersection


def refine_object_overlap(tr1, tr2) :
    """
    Estimate degree of overlap between two trajectory objects.   

    Parameters
    ----------
    tr1 : xarray.Dataset
        Trajectory Dataset.
    tr2 : xarray.Dataset
        Trajectory Dataset.

    Returns
    -------
    float
        Fractional overlap.

    """
    def extract_obj_as1Dint(traj) :

        dx = traj.attrs['dx']
        dy = traj.attrs['dy']
        dz = traj.attrs['dz']

        nx = int(round(traj.attrs['Lx'] / dx))
        ny = int(round(traj.attrs['Ly'] / dy))

        mask = traj.object_mask.values

        if type(mask[0]) is not bool: mask = (mask == 1)

        itrx = (traj.x.values[mask] / dx + 0.5).astype(int)
        itry = (traj.y.values[mask] / dy + 0.5).astype(int)
        itrz = (traj.z.values[mask] / dz + 0.5).astype(int)

        tr1D = np.unique(itrx + nx * (itry + ny * itrz))
        return tr1D

    tr1D  = extract_obj_as1Dint(tr1)
    trm1D = extract_obj_as1Dint(tr2)

    max_size = np.max([np.size(tr1D),np.size(trm1D)])
    if max_size > 0 :
        intersection = np.size(np.intersect1d(tr1D, trm1D)) / max_size
    else :
        intersection = 0
    return intersection

def tr_objects_to_numpy(tr: xr.Dataset, to_gridpoint:bool = False) -> dict:
    """
    Convert trajectory data from xarray.Datset to dictionary.

    Parameters
    ----------
    tr : xr.Dataset
        Contains 'x', 'y', 'z' and 'object_mask' variables, 
        'time' coordinate, 'ref_time' and 'object_label' non-dimensional
        coordinates.
    to_gridpoint : bool, optional
        Convert physical units to grid points by dividing x by dx etc.. 
        The default is False.

    Returns
    -------
    dict
        'xyz': position data as numpy array [3, time, trajectory_number],
        'mask': in-object mask as numpy bool array [time, trajectory_number] ,
        'object_label': Object numbers as numpy array [trajectory_number],
        'nobjects' : int number of objects,
        'ref_time' : reference time,
        'time' : time as 1D numpy array,
        'attrs': tr.attrs,

    """

    arr = []

    for c in 'xyz':
        if to_gridpoint:
            v = tr[c].values / tr.attrs[f'd{c}']
        else:
            v = tr[c].values

        arr.append(v)



    if "object_mask" in tr.variables:
        # mask = np.zeros_like(arr[0])
        # mask[tr.object_mask.values] = 1
        # arr.append(mask)
        # varname += 'm'

        mask = tr.object_mask.values

    arr = np.stack(arr)

    objnum = tr.object_label.values

    nobjects = tr.object_label.attrs['nobjects']

    np_traj = {'xyz': arr,
               'mask': mask,
               'object_label': objnum,
               'nobjects' : nobjects,
               'ref_time' : tr.ref_time.item(),
               'time' : tr.time.values,
               'attrs': tr.attrs,
               }

    return np_traj

def tr_data_at_time(traj:dict, req_time:float):
    """
    Select trajectory data at required time from dict format data.

    Parameters
    ----------
    traj : dict
        Trajectory data.
    req_time : float
        Required time.

    Returns
    -------
    dict
        Output data. Format as per input but time dimension absent.

    """
    ind = np.where(traj['time'] == req_time)[0][0]
    traj_time = {'xyz':traj['xyz'][:, ind, :],
                 'mask': traj['mask'][ind, :],
                 'object_label': traj['object_label'],
                 'nobjects' : traj['nobjects'],
                 'ref_time' : traj['ref_time'],
                 'time' : req_time,
                 'attrs': traj['attrs'],
                }
    return traj_time

def tr_data_obj(traj:dict, iobj:int):
    """
    Select trajectory data from required object from dict format data.
  

    Parameters
    ----------
    traj : dict
        Trajectory data.
    iobj : int
        Required object.

    Returns
    -------
    dict
        Output data. Format as per input but just one object.

        DESCRIPTION.

    """

    m = traj['object_label'] == iobj
    traj_iobj = {'xyz': traj['xyz'][..., m],
                 'mask': traj['mask'][..., m],
                 'ref_time': traj['ref_time'],
                 'time': traj['time'],
                 'object_label':[iobj],
                 'nobjects' : 1,
                 'attrs': traj['attrs'],
                }
    return traj_iobj
