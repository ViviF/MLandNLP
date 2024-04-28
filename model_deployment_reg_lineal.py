from flask import Flask, request
from flask_restx import Api, Resource, fields
import joblib
import pandas as pd
import os
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
api = Api(app, version='1.0', title='Used Vehicle Price Prediction', description='Used Vehicle Price Prediction')

modelo_reg_lin = joblib.load(os.path.dirname(__file__) + '/modelo_regresion_lineal.pkl') 

# Variable dummy para utilizar en el modelo
state_dummy = ['FL', 'OH', 'TX', 'CO', 'ME', 'WA', 'CT', 'CA', 'LA',
       'NY', 'PA', 'SC', 'ND', 'NC', 'GA', 'AZ', 'TN', 'KY',
       'NJ', 'UT', 'IA', 'AL', 'NE', 'IL', 'OK', 'MD', 'NV',
       'WV', 'MI', 'VA', 'WI', 'MA', 'OR', 'IN', 'NM', 'MO',
       'HI', 'KS', 'AR', 'MN', 'MS', 'MT', 'AK', 'VT', 'SD',
       'NH', 'DE', 'ID', 'RI', 'WY', 'DC']

make_dummy = ['Jeep', 'Chevrolet', 'BMW', 'Cadillac', 'Mercedes-Benz', 'Toyota',
       'Buick', 'Dodge', 'Volkswagen', 'GMC', 'Ford', 'Hyundai',
       'Mitsubishi', 'Honda', 'Nissan', 'Mazda', 'Volvo', 'Kia', 'Subaru',
       'Chrysler', 'INFINITI', 'Land', 'Porsche', 'Lexus', 'MINI',
       'Lincoln', 'Audi', 'Ram', 'Mercury', 'Tesla', 'FIAT', 'Acura',
       'Scion', 'Pontiac', 'Jaguar', 'Bentley', 'Suzuki', 'Freightliner']

model_dummy = ['Wrangler', 'Tahoe4WD', 'X5AWD', 'SRXLuxury', '3', 'C-ClassC300',
       'CamryL', 'TacomaPreRunner', 'LaCrosse4dr', 'ChargerSXT',
       'CamryLE', 'Jetta', 'AcadiaFWD', 'EscapeSE', 'SonataLimited',
       'Santa', 'Outlander', 'CruzeSedan', 'Civic', 'CorollaL', '350Z2dr',
       'EdgeSEL', 'F-1502WD', 'FocusSE', 'PatriotSport', 'Accord',
       'MustangGT', 'FusionHybrid', 'ColoradoCrew', 'Wrangler4WD',
       'CR-VEX-L', 'CTS', 'CherokeeLimited', 'Yukon', 'Elantra', 'New',
       'CorollaLE', 'Canyon4WD', 'Golf', 'Sonata4dr', 'Elantra4dr',
       'PatriotLatitude', 'Mazda35dr', 'Tacoma2WD', 'Corolla4dr',
       'Silverado', 'TerrainFWD', 'EscapeFWD', 'Grand', 'RAV4FWD',
       'Liberty4WD', 'FocusTitanium', 'DurangoAWD', 'S60T5', 'CivicLX',
       'MuranoAWD', 'ForteEX', 'TraverseAWD', 'CamaroConvertible',
       'Sportage2WD', 'Pathfinder4WD', 'Highlander4dr', 'WRXSTI', 'Ram',
       'F-150XLT', 'SiennaXLE', 'LaCrosseFWD', 'RogueFWD', 'CamaroCoupe',
       'JourneySXT', 'AccordEX-L', 'Escape4WD', 'OptimaEX', 'FusionSE',
       '5', 'F-150SuperCrew', '200Limited', 'Malibu', 'CompassSport',
       'G37', 'CanyonCrew', 'Malibu1LT', 'MustangPremium', 'MustangBase',
       'Sierra', 'FlexLimited', 'Tahoe2WD', 'Transit', 'Outback2.5i',
       'TucsonLimited', 'Rover', 'CayenneAWD', 'MalibuLT', 'TucsonFWD',
       'F-150FX2', 'Camaro2dr', 'Colorado4WD', 'SonataSE', 'ESES',
       'EnclavePremium', 'CR-VEX', 'F-150STX', 'Impreza', 'EquinoxFWD',
       'Cooper', 'Super', 'Passat4dr', '911', 'CivicEX', 'CamrySE',
       'Highlander4WD', 'Corvette2dr', '200S', 'PilotLX', 'SorentoEX',
       'RioLX', 'ExplorerXLT', 'CorvetteCoupe', 'EnclaveLeather',
       'Avalanche4WD', 'TacomaBase', 'Versa5dr', 'MKXFWD',
       'SL-ClassSL500', 'VeracruzFWD', 'CorollaS', 'PriusTwo', 'CR-V2WD',
       'Lucerne4dr', '4Runner4dr', 'PilotTouring', 'CR-VLX',
       'CompassLatitude', 'Altima4dr', 'OptimaLX', 'Focus5dr',
       'Charger4dr', 'AcadiaAWD', 'JourneyFWD', '7', 'RX', 'MalibuLS',
       'LSLS', 'SportageLX', 'Yukon4WD', 'SorentoLX', 'TiguanSEL',
       'Camry4dr', 'F-1504WD', 'PriusBase', 'AccordLX', 'Q7quattro',
       'ExplorerLimited', '4RunnerSR5', 'OdysseyEX-L', 'C-ClassC',
       'CX-9FWD', 'JourneyAWD', 'Sorento2WD', 'F-250Lariat', 'Prius',
       'TahoeLT', '25004WD', 'Escalade4dr', 'GTI4dr', '4RunnerRWD',
       'FX35AWD', 'XC90T6', 'Taurus4dr', 'AvalonXLE', '300300S', 'G35',
       'F-150Platinum', 'TerrainAWD', 'GXGX', 'MKXAWD', 'Town',
       'CamryXLE', 'VeracruzAWD', 'FusionS', 'Challenger2dr', 'Tundra',
       'Navigator4WD', 'Legacy3.6R', 'GS', 'E-ClassE350', 'Suburban2WD',
       'A44dr', 'RegalTurbo', 'Outback3.6R', '4Runner4WD', 'Legacy2.5i',
       '1', 'Yukon2WD', 'Explorer', 'PilotEX-L', '200LX', 'M-ClassML350',
       'RAV4XLE', 'WranglerSport', 'Model', 'FJ', 'Titan', 'Titan4WD',
       'FlexSEL', 'OdysseyTouring', 'SorentoSX', 'RAV4Base', 'OdysseyEX',
       'Explorer4WD', 'Mustang2dr', 'EdgeLimited', 'FusionSEL',
       'Yukon4dr', 'Touareg4dr', 'Matrix5dr', 'CTCT', 'CherokeeSport',
       '6', 'Maxima4dr', 'Frontier4WD', 'PriusThree', 'F-350XL', '500Pop',
       'RDXAWD', 'Tacoma4WD', 'Optima4dr', 'Q5quattro', 'X3xDrive28i',
       'RDXFWD', 'X5xDrive35i', 'Malibu4dr', 'ExpeditionXLT', 'Ranger2WD',
       'Patriot4WD', 'Quest4dr', 'TaurusSE', 'PathfinderS', 'Murano2WD',
       'LS', 'SiennaLimited', 'ES', 'SiennaLE', 'F-150Lariat', 'Titan2WD',
       'Durango2WD', 'Tahoe4dr', 'Focus4dr', 'YarisBase', 'TaurusLimited',
       'RAV44WD', 'C-Class4dr', 'Soul+', 'TundraBase', 'Expedition',
       'ImpalaLT', 'SedonaLX', 'Sequoia4WD', 'ElantraLimited', '15002WD',
       'Suburban4WD', 'FiestaSE', '15004WD', 'TundraSR5', 'Camry',
       'RAV4Limited', 'RangerSuperCab', 'MDXAWD', 'RAV4LE',
       'ChallengerR/T', 'FlexSE', 'ForteLX', 'TraverseFWD',
       'LibertySport', 'ISIS', 'Impala4dr', 'Tundra4WD', 'F-250XLT',
       'RXRX', 'Armada2WD', 'Frontier', 'WranglerRubicon', 'EquinoxAWD',
       'PilotEX', 'TiguanS', 'EscaladeAWD', 'DTS4dr', 'Pilot2WD',
       'Express', 'PacificaLimited', 'CanyonExtended', 'MX5', 'EscapeS',
       'IS', 'C-ClassC350', 'Compass4WD', 'SportageEX', 'Legacy',
       'E-ClassE', 'Dakota4WD', '300300C', 'Forte', 'SportageAWD',
       'TaurusSEL', 'Xterra4WD', 'GSGS', 'Explorer4dr', 'F-150XL',
       'SportageSX', 'xB5dr', 'TundraLimited', 'CruzeLT', 'Wrangler2dr',
       'HighlanderFWD', 'Sprinter', 'Highlander', 'Prius5dr', 'CX-9Grand',
       'CTS4dr', 'Econoline', 'AccordEX', 'RAV4Sport', '35004WD',
       'ChargerSE', 'OdysseyLX', 'TucsonAWD', 'CX-7FWD', 'AccordLX-S',
       'Navigator4dr', 'EscapeXLT', 'TiguanSE', 'Cayman2dr', 'TaurusSHO',
       'F-150FX4', 'Ranger4WD', 'OptimaSX', 'SequoiaSR5', 'G64dr',
       'HighlanderLimited', 'ExplorerFWD', 'F-350King', 'PriusFive',
       'Yaris4dr', 'PatriotLimited', 'Lancer4dr', 'HighlanderSE',
       'CompassLimited', 'S2000Manual', 'F-250King', 'Forester2.5X',
       'Fusion4dr', 'Frontier2WD', 'FocusST', 'Pathfinder2WD',
       'Sentra4dr', 'XF4dr', 'F-250XL', 'PacificaTouring',
       'MustangDeluxe', 'Caliber4dr', 'GTI2dr', 'Mazda34dr', 'FocusS',
       'Sienna5dr', 'CR-V4WD', 'CX-9Touring', 'Mazda64dr', 'Forester4dr',
       '1500Tradesman', 'MDX4WD', 'Escalade', 'TL4dr', 'CX-9AWD',
       'Canyon2WD', 'A64dr', 'A8', 'Armada4WD', 'Impreza2.0i', 'GX',
       'QX564WD', 'CC4dr', 'MKZ4dr', 'Yaris', 'FitSport', 'Regal4dr',
       'Tundra2WD', 'X3AWD', 'SonicSedan', 'Cobalt4dr', 'RidgelineRTL',
       'CivicSi', 'AvalonLimited', 'XC90FWD', 'Outlander2WD', 'RAV44dr',
       'ColoradoExtended', 'ExpeditionLimited', '3004dr', '200Touring',
       'SC', 'X1xDrive28i', 'SonicHatch', 'GLI4dr', 'PilotSE', 'Savana',
       'RegalPremium', 'CR-VSE', 'RegalGS', 'XC90AWD', 'EdgeSport',
       'PriusFour', 'SiennaSE', '1500Laramie', '300Base', 'Pilot4WD',
       'A34dr', 'HighlanderBase', 'Expedition4WD', 'STS4dr', 'SoulBase',
       'Xterra2WD', 'CT', 'tC2dr', 'Tiguan2WD', 'CR-ZEX', 'MustangShelby',
       'C702dr', 'WranglerX', 'WranglerSahara', 'DurangoSXT',
       'Sequoia4dr', 'Outlander4WD', 'Expedition2WD', 'Navigator',
       '9112dr', 'Vibe4dr', 'F-150King', '300Limited', 'XC60T6',
       'CivicEX-L', 'Avalanche2WD', 'F-350XLT', 'ExplorerBase', 'MuranoS',
       'LXLX', 'EdgeSE', 'ImpalaLS', 'Land', 'E-ClassE320', 'Milan4dr',
       'Boxster2dr', 'RAV4', 'Eos2dr', 'SedonaEX', 'xD5dr', 'Colorado2WD',
       'Monte', 'Escape4dr', 'LX', 'FiestaS', 'F-350Lariat', 'Galant4dr',
       'TT2dr', 'Xterra4dr', 'SequoiaLimited', '4RunnerLimited',
       'Genesis', 'Suburban4dr', 'EnclaveConvenience', 'LaCrosseAWD',
       'Versa4dr', 'Cobalt2dr', 'XC60FWD', 'F-150Limited', 'Dakota2WD',
       'S44dr', '4Runner2WD', 'Sedona4dr', 'RidgelineSport',
       'TSXAutomatic', 'ImprezaSport', 'SLK-ClassSLK350', 'Accent4dr',
       'CorvetteConvertible', 'Avalon4dr', 'Passat', '25002WD',
       'ExplorerEddie', 'LibertyLimited', 'CTS-V', '4RunnerTrail',
       'Eclipse3dr', 'Azera4dr', 'TahoeLS', 'Continental', 'XJ4dr',
       'ForteSX', 'SequoiaPlatinum', 'FocusSEL', 'Durango4dr',
       'CamryBase', 'XC704dr', 'S804dr', 'Element4WD', 'YarisLE',
       'WRXBase', 'TLAutomatic', 'AvalonTouring', 'XK2dr', 'PT',
       'PathfinderSE', '300Touring', 'Navigator2WD', 'XC60AWD',
       'EscapeLimited', 'WRXLimited', 'AccordSE', 'QX562WD',
       'Escalade2WD', 'EscapeLImited', 'PriusOne', 'Element2WD',
       'Excursion137"', 'WRXPremium', 'RX-84dr']

parser = api.parser()
parser.add_argument('Year', type=int, required=True, help='Year of the vehicle', location='args')
parser.add_argument('Mileage', type=int, required=True, help='Mileage of the vehicle', location='args')
parser.add_argument('State', type=str, required=True, help='State of the vehicle', location='args')
parser.add_argument('Make', type=str, required=True, help='Make of the vehicle', location='args')
parser.add_argument('Model', type=str, required=True, help='Model of the vehicle', location='args')

resource_fields = api.model('Resource', {
       'Predicted Price': fields.Float(description='Predicted price of the vehicle')
})

@api.route('/predict')
class PrediccionPrecio(Resource):
    @api.expect(parser)
    @api.marshal_with(resource_fields)
    def post(self):
        args = parser.parse_args()

        # Verificar variables inputs
        year_input = args['Year']
        mileage_input = args['Mileage']
        state_input = args['State']
        make_input = args['Make']
        model_input = args['Model']

        
        # Variables dummies
        state_dummies = pd.DataFrame(False, index=[0], columns=[f'St_{state}' for state in state_dummy])
        print(state_dummies)
        make_dummies = pd.DataFrame(False, index=[0], columns=[f'Mk_{make}' for make in make_dummy])
        model_dummies = pd.DataFrame(False, index=[0], columns=[f'M_{model}' for model in model_dummy])

        column_state = f'St_{state_input}'
        if column_state in state_dummies.columns:
            # Ajustar el valor de la variable dummy
            state_dummies[column_state] = True

        column_make = f'Mk_{make_input}'
        if column_make in make_dummies.columns:
            # Ajustar el valor de la variable dummy
            make_dummies[column_make] = True

        column_model = f'M_{model_input}'
        if column_model in model_dummies.columns:
            # Ajustar el valor de la variable dummy
            model_dummies[column_model] = True

        
        data = {
            'Year': year_input,
            'Mileage': mileage_input
        }
        df = pd.DataFrame(data, index=[0])
        # Crear un DataFrame con los datos de entrada
        #df = pd.DataFrame({'Year': year_input, 'Mileage': mileage_input})
        print(df)
        
        df = pd.concat([df, state_dummies, make_dummies, model_dummies], axis=1)
               
        columnas_reglin = ['Year', 'Mileage', 'St_AK', 'St_AL', 'St_AR', 'St_AZ', 'St_CA', 'St_CO', 'St_CT', 'St_DC', 'St_DE', 'St_FL', 'St_GA', 'St_HI', 'St_IA', 'St_ID', 'St_IL', 'St_IN', 'St_KS', 'St_KY', 'St_LA', 'St_MA', 'St_MD', 'St_ME', 'St_MI', 'St_MN', 'St_MO', 'St_MS', 'St_MT', 'St_NC', 'St_ND', 'St_NE', 'St_NH', 'St_NJ', 'St_NM', 'St_NV', 'St_NY', 'St_OH', 'St_OK', 'St_OR', 'St_PA', 'St_RI', 'St_SC', 'St_SD', 'St_TN', 'St_TX', 'St_UT', 'St_VA', 'St_VT', 'St_WA', 'St_WI', 'St_WV', 'St_WY', 'Mk_Acura', 'Mk_Audi', 'Mk_BMW', 'Mk_Bentley', 'Mk_Buick', 'Mk_Cadillac', 'Mk_Chevrolet', 'Mk_Chrysler', 'Mk_Dodge', 'Mk_FIAT', 'Mk_Ford', 'Mk_Freightliner', 'Mk_GMC', 'Mk_Honda', 'Mk_Hyundai', 'Mk_INFINITI', 'Mk_Jaguar', 'Mk_Jeep', 'Mk_Kia', 'Mk_Land', 'Mk_Lexus', 'Mk_Lincoln', 'Mk_MINI', 'Mk_Mazda', 'Mk_Mercedes-Benz', 'Mk_Mercury', 'Mk_Mitsubishi', 'Mk_Nissan', 'Mk_Pontiac', 'Mk_Porsche', 'Mk_Ram', 'Mk_Scion', 'Mk_Subaru', 'Mk_Suzuki', 'Mk_Tesla', 'Mk_Toyota', 'Mk_Volkswagen', 'Mk_Volvo', 'M_1', 'M_15002WD', 'M_15004WD', 'M_1500Laramie', 'M_1500Tradesman', 'M_200LX', 'M_200Limited', 'M_200S', 'M_200Touring', 'M_25002WD', 'M_25004WD', 'M_3', 'M_300300C', 'M_300300S', 'M_3004dr', 'M_300Base', 'M_300Limited', 'M_300Touring', 'M_35004WD', 'M_350Z2dr', 'M_4Runner2WD', 'M_4Runner4WD', 'M_4Runner4dr', 'M_4RunnerLimited', 'M_4RunnerRWD', 'M_4RunnerSR5', 'M_4RunnerTrail', 'M_5', 'M_500Pop', 'M_6', 'M_7', 'M_911', 'M_9112dr', 'M_A34dr', 'M_A44dr', 'M_A64dr', 'M_A8', 'M_AcadiaAWD', 'M_AcadiaFWD', 'M_Accent4dr', 'M_Accord', 'M_AccordEX', 'M_AccordEX-L', 'M_AccordLX', 'M_AccordLX-S', 'M_AccordSE', 'M_Altima4dr', 'M_Armada2WD', 'M_Armada4WD', 'M_Avalanche2WD', 'M_Avalanche4WD', 'M_Avalon4dr', 'M_AvalonLimited', 'M_AvalonTouring', 'M_AvalonXLE', 'M_Azera4dr', 'M_Boxster2dr', 'M_C-Class4dr', 'M_C-ClassC', 'M_C-ClassC300', 'M_C-ClassC350', 'M_C702dr', 'M_CC4dr', 'M_CR-V2WD', 'M_CR-V4WD', 'M_CR-VEX', 'M_CR-VEX-L', 'M_CR-VLX', 'M_CR-VSE', 'M_CR-ZEX', 'M_CT', 'M_CTCT', 'M_CTS', 'M_CTS-V', 'M_CTS4dr', 'M_CX-7FWD', 'M_CX-9AWD', 'M_CX-9FWD', 'M_CX-9Grand', 'M_CX-9Touring', 'M_Caliber4dr', 'M_Camaro2dr', 'M_CamaroConvertible', 'M_CamaroCoupe', 'M_Camry', 'M_Camry4dr', 'M_CamryBase', 'M_CamryL', 'M_CamryLE', 'M_CamrySE', 'M_CamryXLE', 'M_Canyon2WD', 'M_Canyon4WD', 'M_CanyonCrew', 'M_CanyonExtended', 'M_CayenneAWD', 'M_Cayman2dr', 'M_Challenger2dr', 'M_ChallengerR/T', 'M_Charger4dr', 'M_ChargerSE', 'M_ChargerSXT', 'M_CherokeeLimited', 'M_CherokeeSport', 'M_Civic', 'M_CivicEX', 'M_CivicEX-L', 'M_CivicLX', 'M_CivicSi', 'M_Cobalt2dr', 'M_Cobalt4dr', 'M_Colorado2WD', 'M_Colorado4WD', 'M_ColoradoCrew', 'M_ColoradoExtended', 'M_Compass4WD', 'M_CompassLatitude', 'M_CompassLimited', 'M_CompassSport', 'M_Continental', 'M_Cooper', 'M_Corolla4dr', 'M_CorollaL', 'M_CorollaLE', 'M_CorollaS', 'M_Corvette2dr', 'M_CorvetteConvertible', 'M_CorvetteCoupe', 'M_CruzeLT', 'M_CruzeSedan', 'M_DTS4dr', 'M_Dakota2WD', 'M_Dakota4WD', 'M_Durango2WD', 'M_Durango4dr', 'M_DurangoAWD', 'M_DurangoSXT', 'M_E-ClassE', 'M_E-ClassE320', 'M_E-ClassE350', 'M_ES', 'M_ESES', 'M_Eclipse3dr', 'M_Econoline', 'M_EdgeLimited', 'M_EdgeSE', 'M_EdgeSEL', 'M_EdgeSport', 'M_Elantra', 'M_Elantra4dr', 'M_ElantraLimited', 'M_Element2WD', 'M_Element4WD', 'M_EnclaveConvenience', 'M_EnclaveLeather', 'M_EnclavePremium', 'M_Eos2dr', 'M_EquinoxAWD', 'M_EquinoxFWD', 'M_Escalade', 'M_Escalade2WD', 'M_Escalade4dr', 'M_EscaladeAWD', 'M_Escape4WD', 'M_Escape4dr', 'M_EscapeFWD', 'M_EscapeLImited', 'M_EscapeLimited', 'M_EscapeS', 'M_EscapeSE', 'M_EscapeXLT', 'M_Excursion137"', 'M_Expedition', 'M_Expedition2WD', 'M_Expedition4WD', 'M_ExpeditionLimited', 'M_ExpeditionXLT', 'M_Explorer', 'M_Explorer4WD', 'M_Explorer4dr', 'M_ExplorerBase', 'M_ExplorerEddie', 'M_ExplorerFWD', 'M_ExplorerLimited', 'M_ExplorerXLT', 'M_Express', 'M_F-1502WD', 'M_F-1504WD', 'M_F-150FX2', 'M_F-150FX4', 'M_F-150King', 'M_F-150Lariat', 'M_F-150Limited', 'M_F-150Platinum', 'M_F-150STX', 'M_F-150SuperCrew', 'M_F-150XL', 'M_F-150XLT', 'M_F-250King', 'M_F-250Lariat', 'M_F-250XL', 'M_F-250XLT', 'M_F-350King', 'M_F-350Lariat', 'M_F-350XL', 'M_F-350XLT', 'M_FJ', 'M_FX35AWD', 'M_FiestaS', 'M_FiestaSE', 'M_FitSport', 'M_FlexLimited', 'M_FlexSE', 'M_FlexSEL', 'M_Focus4dr', 'M_Focus5dr', 'M_FocusS', 'M_FocusSE', 'M_FocusSEL', 'M_FocusST', 'M_FocusTitanium', 'M_Forester2.5X', 'M_Forester4dr', 'M_Forte', 'M_ForteEX', 'M_ForteLX', 'M_ForteSX', 'M_Frontier', 'M_Frontier2WD', 'M_Frontier4WD', 'M_Fusion4dr', 'M_FusionHybrid', 'M_FusionS', 'M_FusionSE', 'M_FusionSEL', 'M_G35', 'M_G37', 'M_G64dr', 'M_GLI4dr', 'M_GS', 'M_GSGS', 'M_GTI2dr', 'M_GTI4dr', 'M_GX', 'M_GXGX', 'M_Galant4dr', 'M_Genesis', 'M_Golf', 'M_Grand', 'M_Highlander', 'M_Highlander4WD', 'M_Highlander4dr', 'M_HighlanderBase', 'M_HighlanderFWD', 'M_HighlanderLimited', 'M_HighlanderSE', 'M_IS', 'M_ISIS', 'M_Impala4dr', 'M_ImpalaLS', 'M_ImpalaLT', 'M_Impreza', 'M_Impreza2.0i', 'M_ImprezaSport', 'M_Jetta', 'M_JourneyAWD', 'M_JourneyFWD', 'M_JourneySXT', 'M_LS', 'M_LSLS', 'M_LX', 'M_LXLX', 'M_LaCrosse4dr', 'M_LaCrosseAWD', 'M_LaCrosseFWD', 'M_Lancer4dr', 'M_Land', 'M_Legacy', 'M_Legacy2.5i', 'M_Legacy3.6R', 'M_Liberty4WD', 'M_LibertyLimited', 'M_LibertySport', 'M_Lucerne4dr', 'M_M-ClassML350', 'M_MDX4WD', 'M_MDXAWD', 'M_MKXAWD', 'M_MKXFWD', 'M_MKZ4dr', 'M_MX5', 'M_Malibu', 'M_Malibu1LT', 'M_Malibu4dr', 'M_MalibuLS', 'M_MalibuLT', 'M_Matrix5dr', 'M_Maxima4dr', 'M_Mazda34dr', 'M_Mazda35dr', 'M_Mazda64dr', 'M_Milan4dr', 'M_Model', 'M_Monte', 'M_Murano2WD', 'M_MuranoAWD', 'M_MuranoS', 'M_Mustang2dr', 'M_MustangBase', 'M_MustangDeluxe', 'M_MustangGT', 'M_MustangPremium', 'M_MustangShelby', 'M_Navigator', 'M_Navigator2WD', 'M_Navigator4WD', 'M_Navigator4dr', 'M_New', 'M_OdysseyEX', 'M_OdysseyEX-L', 'M_OdysseyLX', 'M_OdysseyTouring', 'M_Optima4dr', 'M_OptimaEX', 'M_OptimaLX', 'M_OptimaSX', 'M_Outback2.5i', 'M_Outback3.6R', 'M_Outlander', 'M_Outlander2WD', 'M_Outlander4WD', 'M_PT', 'M_PacificaLimited', 'M_PacificaTouring', 'M_Passat', 'M_Passat4dr', 'M_Pathfinder2WD', 'M_Pathfinder4WD', 'M_PathfinderS', 'M_PathfinderSE', 'M_Patriot4WD', 'M_PatriotLatitude', 'M_PatriotLimited', 'M_PatriotSport', 'M_Pilot2WD', 'M_Pilot4WD', 'M_PilotEX', 'M_PilotEX-L', 'M_PilotLX', 'M_PilotSE', 'M_PilotTouring', 'M_Prius', 'M_Prius5dr', 'M_PriusBase', 'M_PriusFive', 'M_PriusFour', 'M_PriusOne', 'M_PriusThree', 'M_PriusTwo', 'M_Q5quattro', 'M_Q7quattro', 'M_QX562WD', 'M_QX564WD', 'M_Quest4dr', 'M_RAV4', 'M_RAV44WD', 'M_RAV44dr', 'M_RAV4Base', 'M_RAV4FWD', 'M_RAV4LE', 'M_RAV4Limited', 'M_RAV4Sport', 'M_RAV4XLE', 'M_RDXAWD', 'M_RDXFWD', 'M_RX', 'M_RX-84dr', 'M_RXRX', 'M_Ram', 'M_Ranger2WD', 'M_Ranger4WD', 'M_RangerSuperCab', 'M_Regal4dr', 'M_RegalGS', 'M_RegalPremium', 'M_RegalTurbo', 'M_RidgelineRTL', 'M_RidgelineSport', 'M_RioLX', 'M_RogueFWD', 'M_Rover', 'M_S2000Manual', 'M_S44dr', 'M_S60T5', 'M_S804dr', 'M_SC', 'M_SL-ClassSL500', 'M_SLK-ClassSLK350', 'M_SRXLuxury', 'M_STS4dr', 'M_Santa', 'M_Savana', 'M_Sedona4dr', 'M_SedonaEX', 'M_SedonaLX', 'M_Sentra4dr', 'M_Sequoia4WD', 'M_Sequoia4dr', 'M_SequoiaLimited', 'M_SequoiaPlatinum', 'M_SequoiaSR5', 'M_Sienna5dr', 'M_SiennaLE', 'M_SiennaLimited', 'M_SiennaSE', 'M_SiennaXLE', 'M_Sierra', 'M_Silverado', 'M_Sonata4dr', 'M_SonataLimited', 'M_SonataSE', 'M_SonicHatch', 'M_SonicSedan', 'M_Sorento2WD', 'M_SorentoEX', 'M_SorentoLX', 'M_SorentoSX', 'M_Soul+', 'M_SoulBase', 'M_Sportage2WD', 'M_SportageAWD', 'M_SportageEX', 'M_SportageLX', 'M_SportageSX', 'M_Sprinter', 'M_Suburban2WD', 'M_Suburban4WD', 'M_Suburban4dr', 'M_Super', 'M_TL4dr', 'M_TLAutomatic', 'M_TSXAutomatic', 'M_TT2dr', 'M_Tacoma2WD', 'M_Tacoma4WD', 'M_TacomaBase', 'M_TacomaPreRunner', 'M_Tahoe2WD', 'M_Tahoe4WD', 'M_Tahoe4dr', 'M_TahoeLS', 'M_TahoeLT', 'M_Taurus4dr', 'M_TaurusLimited', 'M_TaurusSE', 'M_TaurusSEL', 'M_TaurusSHO', 'M_TerrainAWD', 'M_TerrainFWD', 'M_Tiguan2WD', 'M_TiguanS', 'M_TiguanSE', 'M_TiguanSEL', 'M_Titan', 'M_Titan2WD', 'M_Titan4WD', 'M_Touareg4dr', 'M_Town', 'M_Transit', 'M_TraverseAWD', 'M_TraverseFWD', 'M_TucsonAWD', 'M_TucsonFWD', 'M_TucsonLimited', 'M_Tundra', 'M_Tundra2WD', 'M_Tundra4WD', 'M_TundraBase', 'M_TundraLimited', 'M_TundraSR5', 'M_VeracruzAWD', 'M_VeracruzFWD', 'M_Versa4dr', 'M_Versa5dr', 'M_Vibe4dr', 'M_WRXBase', 'M_WRXLimited', 'M_WRXPremium', 'M_WRXSTI', 'M_Wrangler', 'M_Wrangler2dr', 'M_Wrangler4WD', 'M_WranglerRubicon', 'M_WranglerSahara', 'M_WranglerSport', 'M_WranglerX', 'M_X1xDrive28i', 'M_X3AWD', 'M_X3xDrive28i', 'M_X5AWD', 'M_X5xDrive35i', 'M_XC60AWD', 'M_XC60FWD', 'M_XC60T6', 'M_XC704dr', 'M_XC90AWD', 'M_XC90FWD', 'M_XC90T6', 'M_XF4dr', 'M_XJ4dr', 'M_XK2dr', 'M_Xterra2WD', 'M_Xterra4WD', 'M_Xterra4dr', 'M_Yaris', 'M_Yaris4dr', 'M_YarisBase', 'M_YarisLE', 'M_Yukon', 'M_Yukon2WD', 'M_Yukon4WD', 'M_Yukon4dr', 'M_tC2dr', 'M_xB5dr', 'M_xD5dr']

        df = df.reindex(columns=columnas_reglin)
        # Realizar la predicción con el modelo CatBoost
        prediction = modelo_reg_lin.predict(df)[0]
        
        # Devolver el precio estimado
        return {'Predicted Price': prediction}, 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)