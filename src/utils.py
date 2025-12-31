from haversine import haversine, Unit

def calculate_distance(coord1, coord2):
    """
    Calculate the great-circle distance between two points 
    on the Earth using the Haversine formula.
    
    Args:
        coord1 (tuple): (Longitude, Latitude)
        coord2 (tuple): (Longitude, Latitude)
        
    Returns:
        float: Distance in kilometers.
    """
    # The library expects (Lat, Lon), but our data uses (Lon, Lat).
    # We must swap them for the calculation.
    point1 = (coord1[1], coord1[0])
    point2 = (coord2[1], coord2[0])
    
    return haversine(point1, point2, unit=Unit.KILOMETERS)