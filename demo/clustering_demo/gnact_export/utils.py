# db connections
import psycopg2
from sqlalchemy import create_engine
import datetime
import numpy as np
import pandas as pd


# %% Database connection functions
def connect_psycopg2(params, print_verbose=True):
    """ Connect to the PostgreSQL database server using a dict of params. """
    try:
        # connect to the PostgreSQL server
        if print_verbose is True:
            print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
        c = conn.cursor()
        # Execute a statement
        if print_verbose is True:
            print('PostgreSQL database version:')
        c.execute('SELECT version()')
        db_version = c.fetchone()
        if print_verbose is True:
            print(db_version)
        # close the communication with the PostgreSQL
        c.close()
        if print_verbose is True:
            print('Connection created for', params['database'])
        return conn
    except Exception as error:
        print(error)


def connect_engine(params, print_verbose=True):
    """ Create SQLAlchemy engine from dict of parameters"""
    if print_verbose == True:
        print('Creating Engine...')
    try:
        engine = (create_engine(f"postgresql://{params['user']}:{params['password']}"
                                f"@{params['host']}:{params['port']}/{params['database']}"))
        with engine.connect() as conn:
            data = conn.execute('SELECT version()')
        if print_verbose is True:
            print('Engine created for', params['database'])
            for row in data:
                print(row[0])
        return engine
    except Exception as error:
        print('Connection failed.')
        print(error)


def drop_table(table, conn):
    """Simple func to drop tables based on table name and conn"""
    c = conn.cursor()
    c.execute(f'drop table if exists {table} cascade')
    conn.commit()
    c.close()


def make_tables_geom(table, conn, schema_name='public'):
    """A function to make the provided table in provided schema a geometry.
    Table needs to have an existing lat and lon column and not already have a geom column."""
    # add a geom column to the new table and populate it from the lat and lon columns
    c = conn.cursor()
    c.execute(f"""ALTER TABLE {schema_name}.{table} ADD COLUMN
                geom geometry(Point, 4326);""")
    conn.commit()
    c.execute(f"""UPDATE {schema_name}.{table} SET
                geom = ST_SetSRID(ST_MakePoint(lon, lat), 4326);""")
    conn.commit()
    c.close()


def create_schema(schema_name, conn, drop_schema=True, with_date=True):
    """Create the given schamea name, with the option of dropping any
    exisiting schema with the same name.  Can append date to schema."""
    # add the date the run started if desired
    if with_date == True:
        date = str(datetime.date.today()).replace('-', '_')
        schema_name = schema_name + '_' + date

    # if desired, drop existing schema name
    if drop_schema == True:
        c = conn.cursor()
        c.execute("""DROP SCHEMA IF EXISTS {} CASCADE;""".format(schema_name))
        conn.commit()
        print('Old version of schema {} deleted if exists'.format(schema_name))

    # make a new schema to hold the results
    c = conn.cursor()
    c.execute("""CREATE SCHEMA IF NOT EXISTS {};""".format(schema_name))
    conn.commit()
    print('New schema {} created if it did not exist.'.format(schema_name))
    return schema_name

def make_uid_tracker(conn_pg):
    """
    This function makes a tracking table for the UIDs already processed in a script
    :param conn_pg:
    :return: a new table is created (or dropped and recreated) at the end of the conn
    """
    c_pg = conn_pg.cursor()
    c_pg.execute("""DROP TABLE IF EXISTS uid_tracker""")
    conn_pg.commit()
    c_pg.execute("""CREATE TABLE IF NOT EXISTS uid_tracker
    (uid text);""")
    conn_pg.commit()
    c_pg.close()
    print('Clean UID tracker table created.')


def add_to_uid_tracker(uid, conn_pg):
    """
    This function adds a provided uid to the tracking database and returns the len of
    uids already in the table.
    :param uid: tuple, from the returned uid list from the db
    :param conn_pg:
    :return: an int the len of distinct uids in the uid_tracker table
    """
    c_pg = conn_pg.cursor()
    # track completed uids by writing to a new table
    insert_uid_sql = """INSERT INTO uid_tracker (uid) values (%s)"""
    c_pg.execute(insert_uid_sql, uid)
    conn_pg.commit()
    # get total number of uids completed
    c_pg.execute("""SELECT count(distinct(uid)) from uid_tracker""")
    uids_len = (c_pg.fetchone())
    c_pg.close()
    conn_pg.close()
    return uids_len[0]  # remember its a tuple from the db.  [0] gets the int



def calc_dist(df, unit='nm'):
    """
    Takes a df with id, lat, and lon and returns the distance  between
    the previous point to the current point as a series.
    :param df:
    :return:
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [df.lon.shift(1), df.lat.shift(1), df.lon, df.lat])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    r = 2 * np.arcsin(np.sqrt(a))
    if unit == 'mile':
        return 3958.748 * r
    if unit == 'km':
        return 6371 * r
    if unit == 'nm':
        return 3440.65 * r
    else:
        print("Unit is not valid.  Please use 'mile', 'km', or 'nm'.")
        return None


def calc_bearing(df):
    """
    Takes a df with id, lat, and lon and returns the computed bearing between
    the previous point to the current point as a series.
    :param df:
    :return:
    """
    lat1 = np.radians(df.lat.shift(1))
    lat2 = np.radians(df.lat)
    dlon = np.radians(df.lon - df.lon.shift(1))
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
    initial_bearing = np.arctan2(x, y)
    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    initial_bearing = np.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return round(compass_bearing, 2)


def traj_enhance_df(df):
    """
    Takes a df with id, lat, lon, and time.  Returns a df with these same columns as well as
    time_rounded to the lowest minute, time difference from current point and previous group,
    time_diff_hours, course over ground, distance traveled since last point, and speed in knots
    :param df:
    :return:
    """
    # we want to round by minute and drop any duplicates
    df['time_rounded'] = df.time.apply(lambda x: x.floor('min'))
    df.drop_duplicates(['time'], keep='first', inplace=True)
    # calculate time diff between two points
    df['time_diff'] = df.time - df.time.shift(1)
    # time diff in hours needed for speed calc
    df['time_diff_hours'] = pd.to_timedelta(df.time_diff, errors='coerce').dt.total_seconds() / 3600
    # calculate bearing, distance, and speed
    df['cog'] = calc_bearing(df)
    df['dist_nm'] = calc_dist(df, unit='nm')
    df['speed_kts'] = df['dist_nm'] / df['time_diff_hours']
    return df

def build_uid_lists(source_table, target_table, conn):
    """A function that build a list of all unique uids in the source table that are not already
     in the target table.  This can be used to resume processing if interrupted by only continuing
     to process the uids required."""
    print('Building the uid lists...')
    # This list is all of the uids in the table of interest.  It is the
    # total number of uids we will be iterating over.
    c = conn.cursor()
    c.execute(f"""SELECT DISTINCT(uid) FROM {source_table};""")
    uid_list_potential = c.fetchall()
    c.close()

    # if we have to stop the process, we can use the uids that are already completed
    # to build a new list of uids left to complete.  this will allow us to resume
    # processing without repeating any uids.
    c = conn.cursor()
    c.execute(f"""SELECT DISTINCT(uid) FROM {target_table};""")
    uid_list_completed = c.fetchall()
    c.close()

    # find the uids that are not in the edge table yet
    diff = lambda l1, l2: [x for x in l1 if x not in l2]
    uid_list = diff(uid_list_potential, uid_list_completed)
    print('UID lists built.')
    return uid_list