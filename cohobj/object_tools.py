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
            if minflag :
                test1 = (np.min(posid) == 0)
                border = '0'
            else:
                test1 = (np.max(posid) == (n-1))
                border = f"n{['x','y'][dim]}-1"
            if test1 :
                if debug_label :
                    print('Object {:03d} on {}={} border?'.\
                          format(i,['x','y'][dim],border))
                j = i+1
                while j < nobjs :
                    # grid points corresponding to label j
                    posj = np.where(labs == j)
                    posjd = posj[dim]

                    if minflag :
                        test2 = (np.max(posjd) == (n-1))
                        border = f"n{['x','y'][dim]}-1"
                    else:
                        test2 = (np.min(posjd) == 0)
                        border = '0'

                    if test2 :
                        if debug_label :
                            print('Match Object {:03d} on {}={} border?'\
                                  .format(j,['x','y'][dim],border))

                        if minflag :
                            ilist = np.where(posid == 0)
                            jlist = np.where(posjd == (n-1))
                        else :
                            ilist = np.where(posid == (n-1))
                            jlist = np.where(posjd == 0)

                        int1 = np.intersect1d(posi[1-dim][ilist],
                                              posj[1-dim][jlist])
                        # z-intersection
                        int2 = np.intersect1d(posi[2][ilist],
                                              posj[2][jlist])
                        if np.size(int1)>0 and np.size(int2)>0 :
                            if debug_label :
                                print('Yes!',i,j)
                            labs, nobjs = relabel(labs, nobjs, i, j)
                    j += 1
            i += 1
        return labs, nobjs

    labels, nobjects = find_objects_at_edge(True,  0, nx, labels, nobjects)
    labels, nobjects = find_objects_at_edge(False, 0, nx, labels, nobjects)
    labels, nobjects = find_objects_at_edge(True,  1, ny, labels, nobjects)
    labels, nobjects = find_objects_at_edge(False, 1, ny, labels, nobjects)
    
    labels = xr.DataArray(labels, 
                          name='object_labels', 
                          coords=mask.coords, 
                          dims=mask.dims,
                          attrs={'nobjects':nobjects})

    return labels

def mask_to_positions(mask:xr.DataArray)->xr.Dataset:
    """
    Convert 3D logical mask to coordinate positions. 

    Parameters
    ----------
    mask : xr.DataArray
        Evaluates True at required positions.

    Returns
    -------
    positions : xr.Dataset
        Contains data variables "x", "y", "z".
        Coordinates "pos_number" and any others (e.g. "time") in mask.

    """    
    poi = (
        mask.where(mask, drop=True)
            .stack(pos_number=("x", "y", "z"))
            .dropna(dim="pos_number")
    )
    # now we'll turn this 1D dataset where (x, y, z) are coordinates into 
    # one where they are variables instead
    positions = (
        poi.reset_index("pos_number")
           .assign_coords(pos_number=np.arange(poi.pos_number.size))                       
           .reset_coords(["x", "y", "z"])[["x", "y", "z"]]
    )
    
    return positions

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

    Args:
        trajectory     : Array[nt, np, 3] of trajectory points, with nt \
                         times and np points.
        labels         : labels of trajectory points.
        nx,ny   : number of grid points in x and y directions.

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
    
    ds_traj =ds_traj.groupby("object_label").map(_unsplit_object)
        
    return ds_traj
