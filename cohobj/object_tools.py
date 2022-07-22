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

def get_object_labels(mask:xr.DataArray)->xr.DataArray:
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

def unsplit_objects(ds_traj, Lx, Ly) :
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
        and optional "obj_mask" boolean data_var.
    use_mask : bool, optional 
        If true, only use points masked True by "obj_mask". 
        The default is False.

    Returns
    -------
    ds_out : xarray.Dataset
        Contains {x/y/z}_{min/max/mean}

    """
    
    if use_mask:
        ds = ds_traj[["x", "y", "z", "object_label"]].where(ds_traj["obj_mask"]).groupby("object_label")
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
        
    return ds_out
            
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
        
