# Load edge dictionaries from a CSV
# @param csv = filename where the csvs are stored
# Returns a dictionary
def load_edges(csv='edges.stats.csv'):
    edges = []; first = True
    with open(csv,'r') as f:
        for line in f:
            if first:
                first = False
                continue
            line = line.strip()
            data = line.split(',')
            edges.append(dict(ID=data[0],mean=float(data[1]),std=float(data[2]),p50=float(data[3]),p85=float(data[4])))
    return edges

# Loads connections from a CSV
# These are the grid -> map correlations
# @param csv = filename where the csvs are stored
def load_connections(csv='london-seg4.100.conn.csv'):
    connections = []; first = True
    with open(csv,'r') as f:
        for line in f:
            if first:
                first = False
                continue
            data = line.strip().split(',')
            _conn_id = data[0]
            _weight = int(data[1])
            _edges = data[2:]
            connections.append(dict(ID=_conn_id,weight=_weight,edges=_edges))
            
    return connections

# Retrieves the edges that belong to a connection
# @param connection = The connection that the edges belong to.
# @param all_edges = A list of edges to look through
# @return the edges that belong to connection
def get_edges(connection,all_edges):
    edges = []
    for eid in connection['edges']:
        for edge in all_edges:
            if eid == edge['ID']:
                edges.append(edge)
                break
    return edges

# Retrieves the connection that an edge belongs to
# @param edge = the id that needs a parent.
# @param all_connections = a list of all connections to look through
# @return the connection that the edge belonds to
#          or None if nothing is found
def get_connection(edge,all_connections):
    for conn in all_connections:
        if edge['ID'] in conn['edges']:
            return conn
    return None
    
def test():
    edges = load_edges()
    connections = load_connections()
    return
