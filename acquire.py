import env
import pandas as pd
import os

def read_googlesheet(sheet_url):
    '''
   takes in info for google sheets and exports it into a dataframe
    '''
    csv_export_url = sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
    df = pd.read_csv(csv_export_url)
    return df


def get_zillow_all():
    ''' 
    checks for filename (iris_df.csv) in directory and returns that if found
    else it queries for a new one and saves it
    '''
    if os.path.isfile("zillow.csv"):
        df = pd.read_csv("zillow.csv", index_col = 0)
    else:
        sql_query = """
                    SELECT *
                        FROM properties_2017 -- `2,858,627`, "2,985,217"
                            LEFT JOIN airconditioningtype
                                USING (airconditioningtypeid)
                            LEFT JOIN architecturalstyletype
                                USING (architecturalstyletypeid)
                            LEFT JOIN buildingclasstype
                                USING (buildingclasstypeid)
                            LEFT JOIN heatingorsystemtype
                                USING (heatingorsystemtypeid)
                            LEFT JOIN propertylandusetype
                                USING (propertylandusetypeid)
                            LEFT JOIN storytype
                                USING (storytypeid)
                            LEFT JOIN typeconstructiontype
                                USING (typeconstructiontypeid)
                    ;
                    """
        df = pd.read_sql(sql_query,env.get_db_url("zillow"))
        df.to_csv("zillow.csv")
    return df

def get_zillow_single_fam():
    ''' 
    checks for filename (iris_df.csv) in directory and returns that if found
    else it queries for a new one and saves it
    '''
    if os.path.isfile("zillow_single_fam.csv"):
        df = pd.read_csv("zillow_single_fam.csv", index_col = 0)
    else:
        sql_query = """
                    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet,taxvaluedollarcnt,yearbuilt,taxamount,fips
                    FROM properties_2017
                    WHERE properties_2017.propertylandusetypeid = 261
                    ;
                    """
        df = pd.read_sql(sql_query,env.get_db_url("zillow"))
        df.to_csv("zillow_single_fam.csv")
    return df