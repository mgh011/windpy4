#import standar libaries
import os
import re
import datetime
import struct
import numpy as np

#import third part libraries
import pandas as pd
import xarray as xr


#import local libraries



def find_files_by_name_patterns(base_path, pattern1, pattern2='', pattern3=''):
    """
    Returns a list of all files (absolute paths) that contain from one up to
    three specified str patterns in the name.
    
    Parameters
    ----------
    base_path : str
        Path to the main directory (e.g., "/Users/.../TEAMx/raw_data").
    pattern1 : str
        e.g. Name of the station (e.g., "station_1").
    pattern2 : str
        e.g. Name of the station (e.g., "station_1").
    pattern3 : str
        e.g. Name of the station (e.g., "station_1").
    
    Returns
    -------
    matching_files : list of str
        A list of absolute paths to the files that match the specified date.
    """


    matching_files = []
    
    # Check if the station directory actually exists
    if not os.path.isdir(base_path):
        print(f"WARNING: The directory '{base_path}' does not exist or is not valid.")
        #return matching_files
        
    
    # Recursively walk through the base_path
    for root, dirs, files in os.walk(base_path):
        for file in files:
            conditions = []  # Lista per le condizioni da verificare
            
            if pattern1:  # Se pattern1 non è vuoto, lo verifichiamo
                conditions.append(pattern1 in file)
            if pattern2:  # Se pattern2 non è vuoto, lo verifichiamo
                conditions.append(pattern2 in file)
            if pattern3:  # Se pattern3 non è vuoto, lo verifichiamo
                conditions.append(pattern3 in file)
    
            # if conditions are defined, all have to be true
            if all(conditions):  # check if all True
                full_path = os.path.join(root, file)
                matching_files.append(full_path)
    return matching_files

def extract_date(filename):
    """
    Extracts the date from the filename by searching for a pattern in the format: YYYY_MM_DD.
    Returns a datetime object or None if the pattern is not found.
    """
    # Search for the date pattern in the filename
    match = re.search(r'(\d{4}_\d{2}_\d{2})', filename)
    if match:
        date_str = match.group(1)
        # Convert the string to a datetime object
        return datetime.datetime.strptime(date_str, "%Y_%m_%d")
    return None



def read_TOB1(datafile):
    
    def decode_fp2_from_bits(fp2_bits):
        """
        Decodes a 16-bit FP2 binary string into a floating-point number.
        
        Assumes FP2 is stored in **little-endian** byte order.
        
        hhttps://help.campbellsci.com/tx325/shared/formats/fp2.htm?TocPath=Data%20formats%20and%20transmission%20durations%7CPseudobinary%20data%20formats%7C_____1
        """
        # Convert binary string to an integer
        fp2_val = int(fp2_bits, 2)
        
        # Ensure we interpret the value as little-endian
        fp2_val = struct.unpack('<H', struct.pack('>H', fp2_val))[0]  # Swap endianness
    
        # Extract sign bit (bit 16)
        sign = (fp2_val >> 15) & 0x01  # Extract bit 16 (MSB in big-endian, swapped now)
    
        # Extract exponent bits (bits 15-14)
        exponent_bits = (fp2_val >> 13) & 0x03  # Extract bits 15 and 14 (2 bits)
        
        # Extract mantissa (bits 13-0)
        mantissa = fp2_val & 0x1FFF  # Mask the lowest 13 bits (0x1FFF = 8191)
    
        # Handle special cases for exponent == 00
        if exponent_bits == 0:
            if mantissa == 8191:
                return float('inf') if sign == 0 else float('-inf')
            elif mantissa == 8190:
                return float('nan')
    
        # Map exponent bits to actual exponent values
        exponent_map = {
            0b11: -3,  # 11 → 10⁻³
            0b10: -2,  # 10 → 10⁻²
            0b01: -1,  # 01 → 10⁻¹
            0b00:  0   # 00 → 10⁰
        }
        
        exponent = exponent_map[exponent_bits]
    
        # Compute the floating-point FP2 value
        value = ((-1) ** sign) * (10 ** exponent) * mantissa
        return value

    with open(datafile, mode='rb') as rfile:
            content = rfile.read()
        
    # -------------------------------------------------- #
    # ------- READ HEADER & FIELDS & PRECISION --------- #
    # -------------------------------------------------- #
    
    # Find carriage return (CR) & line feed (LF) positions
    lb = [match.span() for match in re.finditer(b'\r\n', content)]
    
    #header1 = content[0:lb[0][0]]
    header2 = content[lb[0][1]:lb[1][0]]
    header3 = content[lb[1][1]:lb[2][0]]
    #header4 = content[lb[2][1]:lb[3][0]]
    header5 = content[lb[3][1]:lb[4][0]]
    
    # get the pos of field names (variables), units, and precision. Use ',' as marker
    # and store them in the lists
    fc2 = [fc.start() for fc in re.finditer(b',', header2)]
    fc3 = [fc.start() for fc in re.finditer(b',', header3)]
    fc5 = [fc.start() for fc in re.finditer(b',', header5)]
    
    fc2.insert(0, -1)
    fc3.insert(0, -1)
    fc5.insert(0, -1)
    
    hvars = []  #variable list
    hunits = [] #unit list
    hprec = [] #precision list
    
    for cnt in range(len(fc2)-1):
        var = header2[fc2[cnt]+2:fc2[cnt+1]-1].decode()
        unit = header3[fc3[cnt]+2:fc3[cnt+1]-1].decode()
        prec = header5[fc5[cnt]+2:fc5[cnt+1]-1].decode()
        hvars.append(var)
        hunits.append(unit)
        hprec.append(prec)
    
    # Handle the last field
    var = header2[fc2[-1]+2:-1].decode()
    unit = header3[fc3[-1]+2:-1].decode()
    prec = header5[fc5[-1]+2:-1].decode()
    hvars.append(var)
    hunits.append(unit)
    hprec.append(prec)
    
        
    # -------------------------------------------------- #
    # ------- MARK THE PRECISION------------------------ #
    # -------------------------------------------------- #
    
    # Build the struct format string and dtype list
    fmt = '<'  # Little-endian
    dtype_list = []
    columns_to_remove = []
    fp2_field_names = [] 
    
    for i, iprec in enumerate(hprec):
        if iprec == 'ULONG':
            fmt += 'L'
            dtype_list.append((hvars[i], 'u4'))
        elif iprec == 'LONG':
            fmt += 'l'
            dtype_list.append((hvars[i], 'i4'))
        elif iprec == 'IEEE4':
            fmt += 'f'
            dtype_list.append((hvars[i], 'f4'))
        elif iprec in ['IEEE8', 'DOUBLE']:
            fmt += 'd'
            dtype_list.append((hvars[i], 'f8'))
        elif 'ASCII' in iprec:
            size = int(re.findall(r'\d+', iprec)[0])
            fmt += f'{size}s'
            dtype_list.append((hvars[i], f'S{size}'))
        elif iprec == 'ASCII(84)':
            fmt += '84x'  # Skip 84 bytes
            columns_to_remove.append(hvars[i])
        elif iprec == 'FP2':
            # For FP2, read 16 bits as an unsigned short
            fmt += 'H'
            """
            IMPORTANT
            NumPy’s dtype system does not recognize a custom type label like 'fp2'. 
            We want to treat those fields as 16‑bit unsigned integers 
            (format 'H' in struct, corresponding to NumPy dtype 'u2') and later convert 
            them (or simply display their bits).
            """
            dtype_list.append((hvars[i], 'u2'))
            fp2_field_names.append(hvars[i]) #keeps track of the fp2, now called u2
        else:
            print(f"Unknown precision: {iprec}. Skipped field.")
            columns_to_remove.append(hvars[i])
            
    
    
    
    sz = struct.calcsize(fmt)
    i1 = lb[4][1]  # Start of the data section.
    nl = (len(content) - i1) // sz
    
    
    # Read the data into a NumPy structured array using the updated dtype_list.
    dtype = np.dtype(dtype_list)
    data_array = np.frombuffer(content[i1:i1+nl*sz], dtype=dtype, count=nl)
    
    # Optionally, convert the structured array to a DataFrame.
    data = pd.DataFrame(data_array)
    for field in fp2_field_names:
        data[field] = data[field].apply(lambda x: decode_fp2_from_bits(format(x, '016b')))
    
    
    data.index.name = 'Time'
    dt = pd.Timestamp('1990-01-01 00:00:00') - pd.Timestamp(0, unit='ns')
    date = pd.DatetimeIndex(data.SECONDS*1e9+data.NANOSECONDS)+dt
    data.index = date
    data.drop(columns=['RECORD', 'SECONDS', 'NANOSECONDS'], inplace=True)
    
    return data


def read_MOMAA_met(datafile, specific_date=None):
    """
    Reads a MOMAA meteorological data file with multiple header lines, filters data for a specific date,
    and returns a pandas DataFrame.

    Parameters
    ----------
    datafile : str
        Path to the MOMAA met data file.
    specific_date : str, optional
        Specific date to filter the data (format: 'YYYY-MM-DD'). If provided, only rows from this date are returned.

    Returns
    -------
    data_df : pandas.DataFrame or None
        DataFrame containing the MOMAA met data with a datetime index.
        Returns None if an error occurs during reading or no data matches the specific_date.
    """
    print(f'Reading MOMAA met data file: {datafile}')
    
    try:
        with open(datafile, 'r') as f:
            # Read all lines from the file
            lines = f.readlines()
        
        # Ensure there are enough lines for headers and data
        if len(lines) < 5:
            raise ValueError(f"File {datafile} does not have enough lines to contain headers and data.")
        
        # Extract column names from the second line (index 1)
        header2 = lines[1].strip().split(',')
        column_names = [col.strip('"') for col in header2]
        
        # Read the data starting from the fifth line (index 4)
        data = pd.read_csv(
            datafile,
            skiprows=4,
            names=column_names,
            na_values=["NAN"],
            parse_dates=['TIMESTAMP'],
            infer_datetime_format=True
        )
        
        # Set 'TIMESTAMP' as the index, if it exists
        if 'TIMESTAMP' in data.columns:
            data.set_index('TIMESTAMP', inplace=True)
        elif 'TS' in data.columns:
            # If the timestamp column is named 'TS' instead of 'TIMESTAMP'
            data['TS'] = pd.to_datetime(data['TS'], infer_datetime_format=True)
            data.set_index('TS', inplace=True)
        else:
            print("No 'TIMESTAMP' or 'TS' column found to set as index.")
        
        # Drop the 'RECORD' column if it exists
        if 'RECORD' in data.columns:
            data.drop(columns=['RECORD'], inplace=True)
        
        # If specific_date is provided, filter the DataFrame
        if specific_date:
            # Convert specific_date to pandas Timestamp
            try:
                date_filter = pd.to_datetime(specific_date).date()
            except Exception as e:
                print(f"Invalid date format for specific_date: {specific_date}. Error: {e}")
                return None
            
            # Filter the DataFrame to include only rows from the specific date
            data = data[data.index.date == date_filter]
            
            if data.empty:
                print(f"No data found for date {specific_date} in file {datafile}.")
                return None
        
        return data
    
    except Exception as e:
        print(f"Error reading file {datafile}: {e}")
        return None




def df_into_ds_hf25(df):
    new_columns = [
    ('pressure', 1), ('pressure', 2),
    ('Ux', 1), ('Uy', 1), ('Uz', 1), ('SonTemp', 1), ('Diag', 1),
    ('Ux', 2), ('Uy', 2), ('Uz', 2), ('SonTemp', 2), ('Diag', 2)
    ]
    df.columns = pd.MultiIndex.from_tuples(new_columns)
    
    # Now df has a MultiIndex on the columns: measurement and height.
    # You have a single "time" index from the DataFrame.
    measurements = ['pressure', 'Ux', 'Uy', 'Uz', 'SonTemp', 'Diag']
    coords = {'time': df.index, 'height': [1, 2]}
    data_vars = {}
    for meas in measurements:
        # For each measurement, extract data corresponding to both heights.
        # df[meas] is a DataFrame with columns corresponding to heights.
        arr = xr.DataArray(df[meas].values, dims=['time', 'height'], coords=coords)
        data_vars[meas] = arr
    
    ds = xr.Dataset(data_vars)
    return ds

def df_into_ds_hf_146(df):
    new_columns = [
        ('pressure', 1),  # solo una quota per pressure
        ('Ux', 1), ('Uy', 1), ('Uz', 1), ('SonTemp', 1), ('Diag', 1),
        ('Ux', 2), ('Uy', 2), ('Uz', 2), ('SonTemp', 2), ('Diag', 2)
    ]
    df.columns = pd.MultiIndex.from_tuples(new_columns)

    measurements = ['pressure', 'Ux', 'Uy', 'Uz', 'SonTemp', 'Diag']
    coords = {'time': df.index, 'height': [1, 2]}
    data_vars = {}

    for meas in measurements:
        if meas == 'pressure':
            # Solo quota 1 presente: crea array con NaN alla quota 2
            n_times = len(df)
            pressure_data = np.full((n_times, 2), np.nan)
            pressure_data[:, 0] = df[meas][1].values  # quota 1
            arr = xr.DataArray(pressure_data, dims=['time', 'height'], coords=coords)
        else:
            arr = xr.DataArray(df[meas].values, dims=['time', 'height'], coords=coords)

        data_vars[meas] = arr

    return xr.Dataset(data_vars)


def df_into_ds(df):

    measurements = ['pressure', 'Ux', 'Uy', 'Uz', 'SonTemp', 'Diag']
    heights = [1, 2]
    coords = {'time': df.index, 'height': heights}
    data_vars = {}

    for meas in measurements:
        n_times = len(df)
        data = np.full((n_times, len(heights)), np.nan)

        for i, h in enumerate(heights):
            if (meas, h) in df.columns:
                data[:, i] = df[(meas, h)].values

        arr = xr.DataArray(data, dims=['time', 'height'], coords=coords)
        data_vars[meas] = arr

    return xr.Dataset(data_vars)


def df_into_ds_hf_seop(df):
    """
    Convert a flat-format DataFrame with sonic and pressure data into a structured xarray.Dataset
    with two vertical levels.

    Assumptions:
    - Time is already set as the DataFrame index.
    - Sonic variables (Ux, Uy, Uz, SonTemp, Diag) may be present on one or both levels.
    - Pressure is taken from the 'digi' column and assumed to exist only at level 2.
    
    Returns:
        xr.Dataset with dimensions ['time', 'height'] and variables:
        'pressure', 'Ux', 'Uy', 'Uz', 'SonTemp', 'Diag'
    """


    # Map flat columns to MultiIndex format
    flat_to_multi = {
        'Ux': ('Ux', 1), 'Uy': ('Uy', 1), 'Uz': ('Uz', 1),
        'SonTemp': ('SonTemp', 1), 'Diag': ('Diag', 1),
        'Ux2': ('Ux', 2), 'Uy2': ('Uy', 2), 'Uz2': ('Uz', 2),
        'SonTemp2': ('SonTemp', 2), 'Diag2': ('Diag', 2),
        'digi': ('pressure', 2)
    }

    # Create a new DataFrame with MultiIndex columns
    df = df.rename(columns=flat_to_multi)
    df = df[[col for col in flat_to_multi.values() if col in df.columns]]
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    measurements = ['pressure', 'Ux', 'Uy', 'Uz', 'SonTemp', 'Diag']
    coords = {'time': df.index, 'height': [1, 2]}
    data_vars = {}

    for meas in measurements:
        n_times = len(df)
        arr_data = np.full((n_times, 2), np.nan)

        if (meas, 1) in df.columns:
            arr_data[:, 0] = df[meas][1].values
        if (meas, 2) in df.columns:
            arr_data[:, 1] = df[meas][2].values

        arr = xr.DataArray(arr_data, dims=['time', 'height'], coords=coords)
        data_vars[meas] = arr

    return xr.Dataset(data_vars)


def df_into_ds_consistent(df):
    import numpy as np
    import pandas as pd
    import xarray as xr

    # Definizione completa di tutte le variabili attese
    measurements = ['pressure', 'Ux', 'Uy', 'Uz', 'SonTemp', 'Diag']
    heights = [1, 2]
    all_keys = [(m, h) for m in measurements for h in heights]

    # Mappa per rinominare colonne piatte → (variabile, quota)
    rename_map = {}
    for col in df.columns:
        if col == "hf_pressure_lvl_1":
            rename_map[col] = ("pressure", 1)
        elif col == "hf_pressure_lvl_2":
            rename_map[col] = ("pressure", 2)
        elif col == "Ux":
            rename_map[col] = ("Ux", 1)
        elif col == "Uy":
            rename_map[col] = ("Uy", 1)
        elif col == "Uz":
            rename_map[col] = ("Uz", 1)
        elif col == "SonTemp":
            rename_map[col] = ("SonTemp", 1)
        elif col == "Diag":
            rename_map[col] = ("Diag", 1)
        elif col == "Ux2":
            rename_map[col] = ("Ux", 2)
        elif col == "Uy2":
            rename_map[col] = ("Uy", 2)
        elif col == "Uz2":
            rename_map[col] = ("Uz", 2)
        elif col == "SonTemp2":
            rename_map[col] = ("SonTemp", 2)
        elif col == "Diag2":
            rename_map[col] = ("Diag", 2)

    # Rinomina solo le colonne mappate e crea un MultiIndex
    df = df.rename(columns=rename_map)
    df = df.loc[:, list(rename_map.values())]  # Scarta colonne non mappate
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    # Costruzione dataset coerente
    coords = {'time': df.index, 'height': heights}
    data_vars = {}

    for meas in measurements:
        n_times = len(df)
        data = np.full((n_times, len(heights)), np.nan)

        for i, h in enumerate(heights):
            key = (meas, h)
            if key in df.columns:
                data[:, i] = df[key].values

        arr = xr.DataArray(data, dims=['time', 'height'], coords=coords)
        data_vars[meas] = arr

    return xr.Dataset(data_vars)


def df_into_ds_consistent(df, invert_sonics=False):
    import numpy as np
    import pandas as pd
    import xarray as xr

    measurements = ['pressure', 'Ux', 'Uy', 'Uz', 'SonTemp', 'Diag']
    heights = [1, 2]

    # Mapping colonne piatte → (variabile, quota)
    rename_map = {}
    for col in df.columns:
        if col == "hf_pressure_lvl_1":
            rename_map[col] = ("pressure", 1)
        elif col == "hf_pressure_lvl_2":
            rename_map[col] = ("pressure", 2)
        elif col == "Ux":
            rename_map[col] = ("Ux", 1)
        elif col == "Uy":
            rename_map[col] = ("Uy", 1)
        elif col == "Uz":
            rename_map[col] = ("Uz", 1)
        elif col == "SonTemp":
            rename_map[col] = ("SonTemp", 1)
        elif col == "Diag":
            rename_map[col] = ("Diag", 1)
        elif col == "Ux2":
            rename_map[col] = ("Ux", 2)
        elif col == "Uy2":
            rename_map[col] = ("Uy", 2)
        elif col == "Uz2":
            rename_map[col] = ("Uz", 2)
        elif col == "SonTemp2":
            rename_map[col] = ("SonTemp", 2)
        elif col == "Diag2":
            rename_map[col] = ("Diag", 2)

    # Rinomina le colonne
    df = df.rename(columns=rename_map)
    df = df.loc[:, list(rename_map.values())]
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    coords = {'time': df.index, 'height': heights}
    data_vars = {}

    for meas in measurements:
        n_times = len(df)
        data = np.full((n_times, len(heights)), np.nan)

        for i, h in enumerate(heights):
            key = (meas, h)
            if key in df.columns:
                data[:, i] = df[key].values

        arr = xr.DataArray(data, dims=['time', 'height'], coords=coords)
        data_vars[meas] = arr

    ds = xr.Dataset(data_vars)

    # ↩️ Inverto i dati sonici se richiesto (stazione 1)
    if invert_sonics:
        sonic_vars = ['Ux', 'Uy', 'Uz', 'SonTemp', 'Diag']
        for var in sonic_vars:
            ds[var].loc[:] = ds[var].sel(height=heights[::-1]).values  # swap quota 1 e 2

    return ds

def df_into_ds_consistent2(
    df,
    invert_sonics: bool = False,
    offsets: dict | None = None,
    station_for_height: dict | None = None,
):
    """
    Converte un DataFrame in un xarray.Dataset con dimensioni (time, height)
    e applica opzionalmente offset per stazione/variabile.

    Parametri
    ---------
    df : pandas.DataFrame
        Colonne attese come nel codice (hf_pressure_lvl_1/2, Ux,Uy,Uz,SonTemp,Diag, e le '2' per il livello 2).
    invert_sonics : bool, default False
        Se True scambia i due livelli solo per le variabili soniche.
    offsets : dict, opzionale
        Dizionario tipo:
        {
          "st1": {"SonTemp": 0.08, "Ux": 0.01, "Uy": -0.0, "Uz": -0.01, "U": 0.0},
          ...
        }
        Gli offset sono bias (measured - reference). Verranno SOTTRATTI ai dati.
    station_for_height : dict, opzionale
        Mappa quota→stazione, es. {1: "st1", 2: "st2"}.
        Se non fornito e offsets è dato, verrà assunto {1: "st1", 2: "st2"}.
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    measurements = ['pressure', 'Ux', 'Uy', 'Uz', 'SonTemp', 'Diag']
    heights = [1, 2]

    # Mapping colonne piatte → (variabile, quota)
    rename_map = {}
    for col in df.columns:
        if col == "hf_pressure_lvl_1":
            rename_map[col] = ("pressure", 1)
        elif col == "hf_pressure_lvl_2":
            rename_map[col] = ("pressure", 2)
        elif col == "Ux":
            rename_map[col] = ("Ux", 1)
        elif col == "Uy":
            rename_map[col] = ("Uy", 1)
        elif col == "Uz":
            rename_map[col] = ("Uz", 1)
        elif col == "SonTemp":
            rename_map[col] = ("SonTemp", 1)
        elif col == "Diag":
            rename_map[col] = ("Diag", 1)
        elif col == "Ux2":
            rename_map[col] = ("Ux", 2)
        elif col == "Uy2":
            rename_map[col] = ("Uy", 2)
        elif col == "Uz2":
            rename_map[col] = ("Uz", 2)
        elif col == "SonTemp2":
            rename_map[col] = ("SonTemp", 2)
        elif col == "Diag2":
            rename_map[col] = ("Diag", 2)

    # Rinomina le colonne e costruisce MultiIndex
    df = df.rename(columns=rename_map)
    df = df.loc[:, list(rename_map.values())]
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    coords = {'time': df.index, 'height': heights}
    data_vars = {}

    for meas in measurements:
        n_times = len(df)
        data = np.full((n_times, len(heights)), np.nan)

        for i, h in enumerate(heights):
            key = (meas, h)
            if key in df.columns:
                data[:, i] = df[key].values

        arr = xr.DataArray(data, dims=['time', 'height'], coords=coords)
        data_vars[meas] = arr

    ds = xr.Dataset(data_vars)

    # 🔧 Applica gli offset (prima di un eventuale invert, perché gli offset sono legati alla stazione fisica)
    if offsets is not None:
        if station_for_height is None:
            station_for_height = {1: "st1", 2: "st2"}  # default sensato, ma personalizzabile

        # Variabili per cui ha senso applicare offset (quelle presenti nel tuo file)
        vars_with_offsets = ['Ux', 'Uy', 'Uz', 'SonTemp', 'U']  # 'U' applicato solo se presente

        for h in heights:
            st = station_for_height.get(h)
            if st is None:
                continue  # nessuna stazione per questa quota

            st_offsets = offsets.get(st, {})
            for var in vars_with_offsets:
                if var in ds and var in st_offsets:
                    # measured_corrected = measured - bias
                    ds[var].loc[dict(height=h)] = ds[var].sel(height=h) - st_offsets[var]

    # ↩️ Inverto i dati sonici se richiesto (swap quota 1 e 2)
    if invert_sonics:
        sonic_vars = ['Ux', 'Uy', 'Uz', 'SonTemp', 'Diag']
        for var in sonic_vars:
            ds[var].loc[:] = ds[var].sel(height=heights[::-1]).values

    return ds


