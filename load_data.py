import numpy as np
from sklearn.preprocessing import scale, normalize

data = [{"Age": "48", "BMI": "23.5", "Glucose": "70", "Resistin": "7.99585", "Classification": "1"},
        {"Age": "83", "BMI": "20.69049454", "Glucose": "92", "Resistin": "4.06405", "Classification": "1"},
        {"Age": "82", "BMI": "23.12467037", "Glucose": "91", "Resistin": "9.27715", "Classification": "1"},
        {"Age": "68", "BMI": "21.36752137", "Glucose": "77", "Resistin": "12.766", "Classification": "1"},
        {"Age": "86", "BMI": "21.11111111", "Glucose": "92", "Resistin": "10.57635", "Classification": "1"},
        {"Age": "49", "BMI": "22.85445769", "Glucose": "92", "Resistin": "10.3176", "Classification": "1"},
        {"Age": "89", "BMI": "22.7", "Glucose": "77", "Resistin": "12.9361", "Classification": "1"},
        {"Age": "76", "BMI": "23.8", "Glucose": "118", "Resistin": "5.1042", "Classification": "1"},
        {"Age": "73", "BMI": "22", "Glucose": "97", "Resistin": "6.28445", "Classification": "1"},
        {"Age": "75", "BMI": "23", "Glucose": "83", "Resistin": "7.0913", "Classification": "1"},
        {"Age": "34", "BMI": "21.47", "Glucose": "78", "Resistin": "6.92", "Classification": "1"},
        {"Age": "29", "BMI": "23.01", "Glucose": "82", "Resistin": "4.58", "Classification": "1"},
        {"Age": "25", "BMI": "22.86", "Glucose": "82", "Resistin": "5.14", "Classification": "1"},
        {"Age": "24", "BMI": "18.67", "Glucose": "88", "Resistin": "6.85", "Classification": "1"},
        {"Age": "38", "BMI": "23.34", "Glucose": "75", "Resistin": "9.35", "Classification": "1"},
        {"Age": "44", "BMI": "20.76", "Glucose": "86", "Resistin": "7.64", "Classification": "1"},
        {"Age": "47", "BMI": "22.03", "Glucose": "84", "Resistin": "3.32", "Classification": "1"},
        {"Age": "61", "BMI": "32.03895937", "Glucose": "85", "Resistin": "13.68392", "Classification": "1"},
        {"Age": "64", "BMI": "34.5297228", "Glucose": "95", "Resistin": "6.70188", "Classification": "1"},
        {"Age": "32", "BMI": "36.51263743", "Glucose": "87", "Resistin": "17.10223", "Classification": "1"},
        {"Age": "36", "BMI": "28.57667585", "Glucose": "86", "Resistin": "9.1539", "Classification": "1"},
        {"Age": "34", "BMI": "31.97501487", "Glucose": "87", "Resistin": "5.62592", "Classification": "1"},
        {"Age": "29", "BMI": "32.27078777", "Glucose": "84", "Resistin": "24.6033", "Classification": "1"},
        {"Age": "35", "BMI": "30.27681661", "Glucose": "84", "Resistin": "16.43706", "Classification": "1"},
        {"Age": "54", "BMI": "30.48315806", "Glucose": "90", "Resistin": "10.19299", "Classification": "1"},
        {"Age": "45", "BMI": "37.03560819", "Glucose": "83", "Resistin": "8.70448", "Classification": "1"},
        {"Age": "50", "BMI": "38.57875854", "Glucose": "106", "Resistin": "11.78388", "Classification": "1"},
        {"Age": "66", "BMI": "31.44654088", "Glucose": "90", "Resistin": "23.3819", "Classification": "1"},
        {"Age": "35", "BMI": "35.2507611", "Glucose": "90", "Resistin": "22.03703", "Classification": "1"},
        {"Age": "36", "BMI": "34.17489", "Glucose": "80", "Resistin": "15.72187", "Classification": "1"},
        {"Age": "66", "BMI": "36.21227888", "Glucose": "101", "Resistin": "22.32024", "Classification": "1"},
        {"Age": "53", "BMI": "36.7901662", "Glucose": "101", "Resistin": "10.26309", "Classification": "1"},
        {"Age": "28", "BMI": "35.85581466", "Glucose": "87", "Resistin": "21.44366", "Classification": "1"},
        {"Age": "43", "BMI": "34.42217362", "Glucose": "89", "Resistin": "6.71026", "Classification": "1"},
        {"Age": "51", "BMI": "27.68877813", "Glucose": "77", "Resistin": "10.37518", "Classification": "1"},
        {"Age": "67", "BMI": "29.60676726", "Glucose": "79", "Resistin": "4.2075", "Classification": "1"},
        {"Age": "66", "BMI": "31.2385898", "Glucose": "82", "Resistin": "3.29175", "Classification": "1"},
        {"Age": "69", "BMI": "35.09270153", "Glucose": "101", "Resistin": "82.1", "Classification": "1"},
        {"Age": "60", "BMI": "26.34929208", "Glucose": "103", "Resistin": "20.2535", "Classification": "1"},
        {"Age": "77", "BMI": "35.58792924", "Glucose": "76", "Resistin": "17.2615", "Classification": "1"},
        {"Age": "76", "BMI": "29.2184076", "Glucose": "83", "Resistin": "8.04375", "Classification": "1"},
        {"Age": "76", "BMI": "27.2", "Glucose": "94", "Resistin": "8.4156", "Classification": "1"},
        {"Age": "75", "BMI": "27.3", "Glucose": "85", "Resistin": "7.5767", "Classification": "1"},
        {"Age": "69", "BMI": "32.5", "Glucose": "93", "Resistin": "11.78796", "Classification": "1"},
        {"Age": "71", "BMI": "30.3", "Glucose": "102", "Resistin": "4.2989", "Classification": "1"},
        {"Age": "66", "BMI": "27.7", "Glucose": "90", "Resistin": "6.7052", "Classification": "1"},
        {"Age": "75", "BMI": "25.7", "Glucose": "94", "Resistin": "4.49685", "Classification": "1"},
        {"Age": "78", "BMI": "25.3", "Glucose": "60", "Resistin": "4.6638", "Classification": "1"},
        {"Age": "69", "BMI": "29.4", "Glucose": "89", "Resistin": "4.53", "Classification": "1"},
        {"Age": "85", "BMI": "26.6", "Glucose": "96", "Resistin": "9.6135", "Classification": "1"},
        {"Age": "76", "BMI": "27.1", "Glucose": "110", "Resistin": "8.49395", "Classification": "1"},
        {"Age": "77", "BMI": "25.9", "Glucose": "85", "Resistin": "11.774", "Classification": "1"},
        {"Age": "45", "BMI": "21.30394858", "Glucose": "102", "Resistin": "23.03408", "Classification": "2"},
        {"Age": "45", "BMI": "20.82999519", "Glucose": "74", "Resistin": "28.0323", "Classification": "2"},
        {"Age": "49", "BMI": "20.9566075", "Glucose": "94", "Resistin": "23.1177", "Classification": "2"},
        {"Age": "34", "BMI": "24.24242424", "Glucose": "92", "Resistin": "12.06534", "Classification": "2"},
        {"Age": "42", "BMI": "21.35991456", "Glucose": "93", "Resistin": "17.37615", "Classification": "2"},
        {"Age": "68", "BMI": "21.08281329", "Glucose": "102", "Resistin": "13.74244", "Classification": "2"},
        {"Age": "51", "BMI": "19.13265306", "Glucose": "93", "Resistin": "5.57055", "Classification": "2"},
        {"Age": "62", "BMI": "22.65625", "Glucose": "92", "Resistin": "10.69548", "Classification": "2"},
        {"Age": "38", "BMI": "22.4996371", "Glucose": "95", "Resistin": "15.73606", "Classification": "2"},
        {"Age": "69", "BMI": "21.51385851", "Glucose": "112", "Resistin": "15.69876", "Classification": "2"},
        {"Age": "49", "BMI": "21.36752137", "Glucose": "78", "Resistin": "22.94254", "Classification": "2"},
        {"Age": "51", "BMI": "22.89281998", "Glucose": "103", "Resistin": "11.55492", "Classification": "2"},
        {"Age": "59", "BMI": "22.83287935", "Glucose": "98", "Resistin": "8.2049", "Classification": "2"},
        {"Age": "45", "BMI": "23.14049587", "Glucose": "116", "Resistin": "5.2633", "Classification": "2"},
        {"Age": "54", "BMI": "24.21875", "Glucose": "86", "Resistin": "10.34455", "Classification": "2"},
        {"Age": "64", "BMI": "22.22222222", "Glucose": "98", "Resistin": "13.91245", "Classification": "2"},
        {"Age": "46", "BMI": "20.83", "Glucose": "88", "Resistin": "13.56", "Classification": "2"},
        {"Age": "44", "BMI": "19.56", "Glucose": "114", "Resistin": "4.62", "Classification": "2"},
        {"Age": "45", "BMI": "20.26", "Glucose": "92", "Resistin": "7.84", "Classification": "2"},
        {"Age": "44", "BMI": "24.74", "Glucose": "106", "Resistin": "5.31", "Classification": "2"},
        {"Age": "51", "BMI": "18.37", "Glucose": "105", "Resistin": "3.21", "Classification": "2"},
        {"Age": "72", "BMI": "23.62", "Glucose": "105", "Resistin": "4.82", "Classification": "2"},
        {"Age": "46", "BMI": "22.21", "Glucose": "86", "Resistin": "5.68", "Classification": "2"},
        {"Age": "43", "BMI": "26.5625", "Glucose": "101", "Resistin": "16.1", "Classification": "2"},
        {"Age": "55", "BMI": "31.97501487", "Glucose": "92", "Resistin": "7.16514", "Classification": "2"},
        {"Age": "43", "BMI": "31.25", "Glucose": "103", "Resistin": "38.6531", "Classification": "2"},
        {"Age": "86", "BMI": "26.66666667", "Glucose": "201", "Resistin": "24.3701", "Classification": "2"},
        {"Age": "41", "BMI": "26.6727633", "Glucose": "97", "Resistin": "27.8325", "Classification": "2"},
        {"Age": "59", "BMI": "28.67262608", "Glucose": "77", "Resistin": "31.6904", "Classification": "2"},
        {"Age": "81", "BMI": "31.64036818", "Glucose": "100", "Resistin": "29.5583", "Classification": "2"},
        {"Age": "48", "BMI": "32.46191136", "Glucose": "99", "Resistin": "10.15726", "Classification": "2"},
        {"Age": "71", "BMI": "25.51020408", "Glucose": "112", "Resistin": "42.7447", "Classification": "2"},
        {"Age": "42", "BMI": "29.296875", "Glucose": "98", "Resistin": "53.6717", "Classification": "2"},
        {"Age": "65", "BMI": "29.666548", "Glucose": "85", "Resistin": "19.46324", "Classification": "2"},
        {"Age": "48", "BMI": "28.125", "Glucose": "90", "Resistin": "16.11032", "Classification": "2"},
        {"Age": "85", "BMI": "27.68877813", "Glucose": "196", "Resistin": "55.2153", "Classification": "2"},
        {"Age": "48", "BMI": "31.25", "Glucose": "199", "Resistin": "53.6308", "Classification": "2"},
        {"Age": "58", "BMI": "29.15451895", "Glucose": "139", "Resistin": "13.97399", "Classification": "2"},
        {"Age": "40", "BMI": "30.83653053", "Glucose": "128", "Resistin": "17.55503", "Classification": "2"},
        {"Age": "82", "BMI": "31.21748179", "Glucose": "100", "Resistin": "19.94687", "Classification": "2"},
        {"Age": "52", "BMI": "30.8012487", "Glucose": "87", "Resistin": "24.24591", "Classification": "2"},
        {"Age": "49", "BMI": "32.46191136", "Glucose": "134", "Resistin": "5.768", "Classification": "2"},
        {"Age": "60", "BMI": "31.23140988", "Glucose": "131", "Resistin": "11.50005", "Classification": "2"},
        {"Age": "49", "BMI": "29.77777778", "Glucose": "70", "Resistin": "20.76801", "Classification": "2"},
        {"Age": "44", "BMI": "27.88761707", "Glucose": "99", "Resistin": "23.03306", "Classification": "2"},
        {"Age": "40", "BMI": "27.63605442", "Glucose": "103", "Resistin": "26.0136", "Classification": "2"},
        {"Age": "71", "BMI": "27.91551882", "Glucose": "104", "Resistin": "49.24184", "Classification": "2"},
        {"Age": "69", "BMI": "28.44444444", "Glucose": "108", "Resistin": "16.48508", "Classification": "2"},
        {"Age": "74", "BMI": "28.65013774", "Glucose": "88", "Resistin": "18.35574", "Classification": "2"},
        {"Age": "66", "BMI": "26.5625", "Glucose": "89", "Resistin": "14.91922", "Classification": "2"},
        {"Age": "65", "BMI": "30.91557669", "Glucose": "97", "Resistin": "20.4685", "Classification": "2"},
        {"Age": "72", "BMI": "29.13631634", "Glucose": "83", "Resistin": "14.76966", "Classification": "2"},
        {"Age": "57", "BMI": "34.83814777", "Glucose": "95", "Resistin": "9.9542", "Classification": "2"},
        {"Age": "73", "BMI": "37.109375", "Glucose": "134", "Resistin": "6.89235", "Classification": "2"},
        {"Age": "45", "BMI": "29.38475666", "Glucose": "90", "Resistin": "15.55625", "Classification": "2"},
        {"Age": "46", "BMI": "33.18", "Glucose": "92", "Resistin": "8.89", "Classification": "2"},
        {"Age": "68", "BMI": "35.56", "Glucose": "131", "Resistin": "4.19", "Classification": "2"},
        {"Age": "75", "BMI": "30.48", "Glucose": "152", "Resistin": "11.73", "Classification": "2"},
        {"Age": "54", "BMI": "36.05", "Glucose": "119", "Resistin": "5.06", "Classification": "2"},
        {"Age": "45", "BMI": "26.85", "Glucose": "92", "Resistin": "10.96", "Classification": "2"},
        {"Age": "62", "BMI": "26.84", "Glucose": "100", "Resistin": "7.32", "Classification": "2"},
        {"Age": "65", "BMI": "32.05", "Glucose": "97", "Resistin": "10.33", "Classification": "2"},
        {"Age": "72", "BMI": "25.59", "Glucose": "82", "Resistin": "3.27", "Classification": "2"},
        {"Age": "86", "BMI": "27.18", "Glucose": "138", "Resistin": "4.35", "Classification": "2"}]
data_all_vars = [
    {'Age': '48', 'BMI': '23.5', 'Glucose': '70', 'Insulin': '2.707', 'HOMA': '0.467408667', 'Leptin': '8.8071',
     'Adiponectin': '9.7024', 'Resistin': '7.99585', 'MCP.1': '417.114', 'Classification': '1'},
    {'Age': '83', 'BMI': '20.69049454', 'Glucose': '92', 'Insulin': '3.115', 'HOMA': '0.706897333', 'Leptin': '8.8438',
     'Adiponectin': '5.429285', 'Resistin': '4.06405', 'MCP.1': '468.786', 'Classification': '1'},
    {'Age': '82', 'BMI': '23.12467037', 'Glucose': '91', 'Insulin': '4.498', 'HOMA': '1.009651067', 'Leptin': '17.9393',
     'Adiponectin': '22.43204', 'Resistin': '9.27715', 'MCP.1': '554.697', 'Classification': '1'},
    {'Age': '68', 'BMI': '21.36752137', 'Glucose': '77', 'Insulin': '3.226', 'HOMA': '0.612724933', 'Leptin': '9.8827',
     'Adiponectin': '7.16956', 'Resistin': '12.766', 'MCP.1': '928.22', 'Classification': '1'},
    {'Age': '86', 'BMI': '21.11111111', 'Glucose': '92', 'Insulin': '3.549', 'HOMA': '0.8053864', 'Leptin': '6.6994',
     'Adiponectin': '4.81924', 'Resistin': '10.57635', 'MCP.1': '773.92', 'Classification': '1'},
    {'Age': '49', 'BMI': '22.85445769', 'Glucose': '92', 'Insulin': '3.226', 'HOMA': '0.732086933', 'Leptin': '6.8317',
     'Adiponectin': '13.67975', 'Resistin': '10.3176', 'MCP.1': '530.41', 'Classification': '1'},
    {'Age': '89', 'BMI': '22.7', 'Glucose': '77', 'Insulin': '4.69', 'HOMA': '0.890787333', 'Leptin': '6.964',
     'Adiponectin': '5.589865', 'Resistin': '12.9361', 'MCP.1': '1256.083', 'Classification': '1'},
    {'Age': '76', 'BMI': '23.8', 'Glucose': '118', 'Insulin': '6.47', 'HOMA': '1.883201333', 'Leptin': '4.311',
     'Adiponectin': '13.25132', 'Resistin': '5.1042', 'MCP.1': '280.694', 'Classification': '1'},
    {'Age': '73', 'BMI': '22', 'Glucose': '97', 'Insulin': '3.35', 'HOMA': '0.801543333', 'Leptin': '4.47',
     'Adiponectin': '10.358725', 'Resistin': '6.28445', 'MCP.1': '136.855', 'Classification': '1'},
    {'Age': '75', 'BMI': '23', 'Glucose': '83', 'Insulin': '4.952', 'HOMA': '1.013839467', 'Leptin': '17.127',
     'Adiponectin': '11.57899', 'Resistin': '7.0913', 'MCP.1': '318.302', 'Classification': '1'},
    {'Age': '34', 'BMI': '21.47', 'Glucose': '78', 'Insulin': '3.469', 'HOMA': '0.6674356', 'Leptin': '14.57',
     'Adiponectin': '13.11', 'Resistin': '6.92', 'MCP.1': '354.6', 'Classification': '1'},
    {'Age': '29', 'BMI': '23.01', 'Glucose': '82', 'Insulin': '5.663', 'HOMA': '1.145436133', 'Leptin': '35.59',
     'Adiponectin': '26.72', 'Resistin': '4.58', 'MCP.1': '174.8', 'Classification': '1'},
    {'Age': '25', 'BMI': '22.86', 'Glucose': '82', 'Insulin': '4.09', 'HOMA': '0.827270667', 'Leptin': '20.45',
     'Adiponectin': '23.67', 'Resistin': '5.14', 'MCP.1': '313.73', 'Classification': '1'},
    {'Age': '24', 'BMI': '18.67', 'Glucose': '88', 'Insulin': '6.107', 'HOMA': '1.33', 'Leptin': '8.88',
     'Adiponectin': '36.06', 'Resistin': '6.85', 'MCP.1': '632.22', 'Classification': '1'},
    {'Age': '38', 'BMI': '23.34', 'Glucose': '75', 'Insulin': '5.782', 'HOMA': '1.06967', 'Leptin': '15.26',
     'Adiponectin': '17.95', 'Resistin': '9.35', 'MCP.1': '165.02', 'Classification': '1'},
    {'Age': '44', 'BMI': '20.76', 'Glucose': '86', 'Insulin': '7.553', 'HOMA': '1.6', 'Leptin': '14.09',
     'Adiponectin': '20.32', 'Resistin': '7.64', 'MCP.1': '63.61', 'Classification': '1'},
    {'Age': '47', 'BMI': '22.03', 'Glucose': '84', 'Insulin': '2.869', 'HOMA': '0.59', 'Leptin': '26.65',
     'Adiponectin': '38.04', 'Resistin': '3.32', 'MCP.1': '191.72', 'Classification': '1'},
    {'Age': '61', 'BMI': '32.03895937', 'Glucose': '85', 'Insulin': '18.077', 'HOMA': '3.790144333',
     'Leptin': '30.7729', 'Adiponectin': '7.780255', 'Resistin': '13.68392', 'MCP.1': '444.395', 'Classification': '1'},
    {'Age': '64', 'BMI': '34.5297228', 'Glucose': '95', 'Insulin': '4.427', 'HOMA': '1.037393667', 'Leptin': '21.2117',
     'Adiponectin': '5.46262', 'Resistin': '6.70188', 'MCP.1': '252.449', 'Classification': '1'},
    {'Age': '32', 'BMI': '36.51263743', 'Glucose': '87', 'Insulin': '14.026', 'HOMA': '3.0099796', 'Leptin': '49.3727',
     'Adiponectin': '5.1', 'Resistin': '17.10223', 'MCP.1': '588.46', 'Classification': '1'},
    {'Age': '36', 'BMI': '28.57667585', 'Glucose': '86', 'Insulin': '4.345', 'HOMA': '0.921719333', 'Leptin': '15.1248',
     'Adiponectin': '8.6', 'Resistin': '9.1539', 'MCP.1': '534.224', 'Classification': '1'},
    {'Age': '34', 'BMI': '31.97501487', 'Glucose': '87', 'Insulin': '4.53', 'HOMA': '0.972138', 'Leptin': '28.7502',
     'Adiponectin': '7.64276', 'Resistin': '5.62592', 'MCP.1': '572.783', 'Classification': '1'},
    {'Age': '29', 'BMI': '32.27078777', 'Glucose': '84', 'Insulin': '5.81', 'HOMA': '1.203832', 'Leptin': '45.6196',
     'Adiponectin': '6.209635', 'Resistin': '24.6033', 'MCP.1': '904.981', 'Classification': '1'},
    {'Age': '35', 'BMI': '30.27681661', 'Glucose': '84', 'Insulin': '4.376', 'HOMA': '0.9067072', 'Leptin': '39.2134',
     'Adiponectin': '9.048185', 'Resistin': '16.43706', 'MCP.1': '733.797', 'Classification': '1'},
    {'Age': '54', 'BMI': '30.48315806', 'Glucose': '90', 'Insulin': '5.537', 'HOMA': '1.229214', 'Leptin': '12.331',
     'Adiponectin': '9.73138', 'Resistin': '10.19299', 'MCP.1': '1227.91', 'Classification': '1'},
    {'Age': '45', 'BMI': '37.03560819', 'Glucose': '83', 'Insulin': '6.76', 'HOMA': '1.383997333', 'Leptin': '39.9802',
     'Adiponectin': '4.617125', 'Resistin': '8.70448', 'MCP.1': '586.173', 'Classification': '1'},
    {'Age': '50', 'BMI': '38.57875854', 'Glucose': '106', 'Insulin': '6.703', 'HOMA': '1.752611067',
     'Leptin': '46.6401', 'Adiponectin': '4.667645', 'Resistin': '11.78388', 'MCP.1': '887.16', 'Classification': '1'},
    {'Age': '66', 'BMI': '31.44654088', 'Glucose': '90', 'Insulin': '9.245', 'HOMA': '2.05239', 'Leptin': '45.9624',
     'Adiponectin': '10.35526', 'Resistin': '23.3819', 'MCP.1': '1102.11', 'Classification': '1'},
    {'Age': '35', 'BMI': '35.2507611', 'Glucose': '90', 'Insulin': '6.817', 'HOMA': '1.513374', 'Leptin': '50.6094',
     'Adiponectin': '6.966895', 'Resistin': '22.03703', 'MCP.1': '667.928', 'Classification': '1'},
    {'Age': '36', 'BMI': '34.17489', 'Glucose': '80', 'Insulin': '6.59', 'HOMA': '1.300426667', 'Leptin': '10.2809',
     'Adiponectin': '5.065915', 'Resistin': '15.72187', 'MCP.1': '581.313', 'Classification': '1'},
    {'Age': '66', 'BMI': '36.21227888', 'Glucose': '101', 'Insulin': '15.533', 'HOMA': '3.869788067',
     'Leptin': '74.7069', 'Adiponectin': '7.53955', 'Resistin': '22.32024', 'MCP.1': '864.968', 'Classification': '1'},
    {'Age': '53', 'BMI': '36.7901662', 'Glucose': '101', 'Insulin': '10.175', 'HOMA': '2.534931667',
     'Leptin': '27.1841', 'Adiponectin': '20.03', 'Resistin': '10.26309', 'MCP.1': '695.754', 'Classification': '1'},
    {'Age': '28', 'BMI': '35.85581466', 'Glucose': '87', 'Insulin': '8.576', 'HOMA': '1.8404096', 'Leptin': '68.5102',
     'Adiponectin': '4.7942', 'Resistin': '21.44366', 'MCP.1': '358.624', 'Classification': '1'},
    {'Age': '43', 'BMI': '34.42217362', 'Glucose': '89', 'Insulin': '23.194', 'HOMA': '5.091856133',
     'Leptin': '31.2128', 'Adiponectin': '8.300955', 'Resistin': '6.71026', 'MCP.1': '960.246', 'Classification': '1'},
    {'Age': '51', 'BMI': '27.68877813', 'Glucose': '77', 'Insulin': '3.855', 'HOMA': '0.732193', 'Leptin': '20.092',
     'Adiponectin': '3.19209', 'Resistin': '10.37518', 'MCP.1': '473.859', 'Classification': '1'},
    {'Age': '67', 'BMI': '29.60676726', 'Glucose': '79', 'Insulin': '5.819', 'HOMA': '1.133929133', 'Leptin': '21.9033',
     'Adiponectin': '2.19428', 'Resistin': '4.2075', 'MCP.1': '585.307', 'Classification': '1'},
    {'Age': '66', 'BMI': '31.2385898', 'Glucose': '82', 'Insulin': '4.181', 'HOMA': '0.845676933', 'Leptin': '16.2247',
     'Adiponectin': '4.267105', 'Resistin': '3.29175', 'MCP.1': '634.602', 'Classification': '1'},
    {'Age': '69', 'BMI': '35.09270153', 'Glucose': '101', 'Insulin': '5.646', 'HOMA': '1.4066068', 'Leptin': '83.4821',
     'Adiponectin': '6.796985', 'Resistin': '82.1', 'MCP.1': '263.499', 'Classification': '1'},
    {'Age': '60', 'BMI': '26.34929208', 'Glucose': '103', 'Insulin': '5.138', 'HOMA': '1.305394533',
     'Leptin': '24.2998', 'Adiponectin': '2.19428', 'Resistin': '20.2535', 'MCP.1': '378.996', 'Classification': '1'},
    {'Age': '77', 'BMI': '35.58792924', 'Glucose': '76', 'Insulin': '3.881', 'HOMA': '0.727558133', 'Leptin': '21.7863',
     'Adiponectin': '8.12555', 'Resistin': '17.2615', 'MCP.1': '618.272', 'Classification': '1'},
    {'Age': '76', 'BMI': '29.2184076', 'Glucose': '83', 'Insulin': '5.376', 'HOMA': '1.1006464', 'Leptin': '28.562',
     'Adiponectin': '7.36996', 'Resistin': '8.04375', 'MCP.1': '698.789', 'Classification': '1'},
    {'Age': '76', 'BMI': '27.2', 'Glucose': '94', 'Insulin': '14.07', 'HOMA': '3.262364', 'Leptin': '35.891',
     'Adiponectin': '9.34663', 'Resistin': '8.4156', 'MCP.1': '377.227', 'Classification': '1'},
    {'Age': '75', 'BMI': '27.3', 'Glucose': '85', 'Insulin': '5.197', 'HOMA': '1.089637667', 'Leptin': '10.39',
     'Adiponectin': '9.000805', 'Resistin': '7.5767', 'MCP.1': '335.393', 'Classification': '1'},
    {'Age': '69', 'BMI': '32.5', 'Glucose': '93', 'Insulin': '5.43', 'HOMA': '1.245642', 'Leptin': '15.145',
     'Adiponectin': '11.78796', 'Resistin': '11.78796', 'MCP.1': '270.142', 'Classification': '1'},
    {'Age': '71', 'BMI': '30.3', 'Glucose': '102', 'Insulin': '8.34', 'HOMA': '2.098344', 'Leptin': '56.502',
     'Adiponectin': '8.13', 'Resistin': '4.2989', 'MCP.1': '200.976', 'Classification': '1'},
    {'Age': '66', 'BMI': '27.7', 'Glucose': '90', 'Insulin': '6.042', 'HOMA': '1.341324', 'Leptin': '24.846',
     'Adiponectin': '7.652055', 'Resistin': '6.7052', 'MCP.1': '225.88', 'Classification': '1'},
    {'Age': '75', 'BMI': '25.7', 'Glucose': '94', 'Insulin': '8.079', 'HOMA': '1.8732508', 'Leptin': '65.926',
     'Adiponectin': '3.74122', 'Resistin': '4.49685', 'MCP.1': '206.802', 'Classification': '1'},
    {'Age': '78', 'BMI': '25.3', 'Glucose': '60', 'Insulin': '3.508', 'HOMA': '0.519184', 'Leptin': '6.633',
     'Adiponectin': '10.567295', 'Resistin': '4.6638', 'MCP.1': '209.749', 'Classification': '1'},
    {'Age': '69', 'BMI': '29.4', 'Glucose': '89', 'Insulin': '10.704', 'HOMA': '2.3498848', 'Leptin': '45.272',
     'Adiponectin': '8.2863', 'Resistin': '4.53', 'MCP.1': '215.769', 'Classification': '1'},
    {'Age': '85', 'BMI': '26.6', 'Glucose': '96', 'Insulin': '4.462', 'HOMA': '1.0566016', 'Leptin': '7.85',
     'Adiponectin': '7.9317', 'Resistin': '9.6135', 'MCP.1': '232.006', 'Classification': '1'},
    {'Age': '76', 'BMI': '27.1', 'Glucose': '110', 'Insulin': '26.211', 'HOMA': '7.111918', 'Leptin': '21.778',
     'Adiponectin': '4.935635', 'Resistin': '8.49395', 'MCP.1': '45.843', 'Classification': '1'},
    {'Age': '77', 'BMI': '25.9', 'Glucose': '85', 'Insulin': '4.58', 'HOMA': '0.960273333', 'Leptin': '13.74',
     'Adiponectin': '9.75326', 'Resistin': '11.774', 'MCP.1': '488.829', 'Classification': '1'},
    {'Age': '45', 'BMI': '21.30394858', 'Glucose': '102', 'Insulin': '13.852', 'HOMA': '3.4851632', 'Leptin': '7.6476',
     'Adiponectin': '21.056625', 'Resistin': '23.03408', 'MCP.1': '552.444', 'Classification': '2'},
    {'Age': '45', 'BMI': '20.82999519', 'Glucose': '74', 'Insulin': '4.56', 'HOMA': '0.832352', 'Leptin': '7.7529',
     'Adiponectin': '8.237405', 'Resistin': '28.0323', 'MCP.1': '382.955', 'Classification': '2'},
    {'Age': '49', 'BMI': '20.9566075', 'Glucose': '94', 'Insulin': '12.305', 'HOMA': '2.853119333', 'Leptin': '11.2406',
     'Adiponectin': '8.412175', 'Resistin': '23.1177', 'MCP.1': '573.63', 'Classification': '2'},
    {'Age': '34', 'BMI': '24.24242424', 'Glucose': '92', 'Insulin': '21.699', 'HOMA': '4.9242264', 'Leptin': '16.7353',
     'Adiponectin': '21.823745', 'Resistin': '12.06534', 'MCP.1': '481.949', 'Classification': '2'},
    {'Age': '42', 'BMI': '21.35991456', 'Glucose': '93', 'Insulin': '2.999', 'HOMA': '0.6879706', 'Leptin': '19.0826',
     'Adiponectin': '8.462915', 'Resistin': '17.37615', 'MCP.1': '321.919', 'Classification': '2'},
    {'Age': '68', 'BMI': '21.08281329', 'Glucose': '102', 'Insulin': '6.2', 'HOMA': '1.55992', 'Leptin': '9.6994',
     'Adiponectin': '8.574655', 'Resistin': '13.74244', 'MCP.1': '448.799', 'Classification': '2'},
    {'Age': '51', 'BMI': '19.13265306', 'Glucose': '93', 'Insulin': '4.364', 'HOMA': '1.0011016', 'Leptin': '11.0816',
     'Adiponectin': '5.80762', 'Resistin': '5.57055', 'MCP.1': '90.6', 'Classification': '2'},
    {'Age': '62', 'BMI': '22.65625', 'Glucose': '92', 'Insulin': '3.482', 'HOMA': '0.790181867', 'Leptin': '9.8648',
     'Adiponectin': '11.236235', 'Resistin': '10.69548', 'MCP.1': '703.973', 'Classification': '2'},
    {'Age': '38', 'BMI': '22.4996371', 'Glucose': '95', 'Insulin': '5.261', 'HOMA': '1.232827667', 'Leptin': '8.438',
     'Adiponectin': '4.77192', 'Resistin': '15.73606', 'MCP.1': '199.055', 'Classification': '2'},
    {'Age': '69', 'BMI': '21.51385851', 'Glucose': '112', 'Insulin': '6.683', 'HOMA': '1.846290133', 'Leptin': '32.58',
     'Adiponectin': '4.138025', 'Resistin': '15.69876', 'MCP.1': '713.239', 'Classification': '2'},
    {'Age': '49', 'BMI': '21.36752137', 'Glucose': '78', 'Insulin': '2.64', 'HOMA': '0.507936', 'Leptin': '6.3339',
     'Adiponectin': '3.886145', 'Resistin': '22.94254', 'MCP.1': '737.672', 'Classification': '2'},
    {'Age': '51', 'BMI': '22.89281998', 'Glucose': '103', 'Insulin': '2.74', 'HOMA': '0.696142667', 'Leptin': '8.0163',
     'Adiponectin': '9.349775', 'Resistin': '11.55492', 'MCP.1': '359.232', 'Classification': '2'},
    {'Age': '59', 'BMI': '22.83287935', 'Glucose': '98', 'Insulin': '6.862', 'HOMA': '1.658774133', 'Leptin': '14.9037',
     'Adiponectin': '4.230105', 'Resistin': '8.2049', 'MCP.1': '355.31', 'Classification': '2'},
    {'Age': '45', 'BMI': '23.14049587', 'Glucose': '116', 'Insulin': '4.902', 'HOMA': '1.4026256', 'Leptin': '17.9973',
     'Adiponectin': '4.294705', 'Resistin': '5.2633', 'MCP.1': '518.586', 'Classification': '2'},
    {'Age': '54', 'BMI': '24.21875', 'Glucose': '86', 'Insulin': '3.73', 'HOMA': '0.791257333', 'Leptin': '8.6874',
     'Adiponectin': '3.70523', 'Resistin': '10.34455', 'MCP.1': '635.049', 'Classification': '2'},
    {'Age': '64', 'BMI': '22.22222222', 'Glucose': '98', 'Insulin': '5.7', 'HOMA': '1.37788', 'Leptin': '12.1905',
     'Adiponectin': '4.783985', 'Resistin': '13.91245', 'MCP.1': '395.976', 'Classification': '2'},
    {'Age': '46', 'BMI': '20.83', 'Glucose': '88', 'Insulin': '3.42', 'HOMA': '0.742368', 'Leptin': '12.87',
     'Adiponectin': '18.55', 'Resistin': '13.56', 'MCP.1': '301.21', 'Classification': '2'},
    {'Age': '44', 'BMI': '19.56', 'Glucose': '114', 'Insulin': '15.89', 'HOMA': '4.468268', 'Leptin': '13.08',
     'Adiponectin': '20.37', 'Resistin': '4.62', 'MCP.1': '220.66', 'Classification': '2'},
    {'Age': '45', 'BMI': '20.26', 'Glucose': '92', 'Insulin': '3.44', 'HOMA': '0.780650667', 'Leptin': '7.65',
     'Adiponectin': '16.67', 'Resistin': '7.84', 'MCP.1': '193.87', 'Classification': '2'},
    {'Age': '44', 'BMI': '24.74', 'Glucose': '106', 'Insulin': '58.46', 'HOMA': '15.28534133', 'Leptin': '18.16',
     'Adiponectin': '16.1', 'Resistin': '5.31', 'MCP.1': '244.75', 'Classification': '2'},
    {'Age': '51', 'BMI': '18.37', 'Glucose': '105', 'Insulin': '6.03', 'HOMA': '1.56177', 'Leptin': '9.62',
     'Adiponectin': '12.76', 'Resistin': '3.21', 'MCP.1': '513.66', 'Classification': '2'},
    {'Age': '72', 'BMI': '23.62', 'Glucose': '105', 'Insulin': '4.42', 'HOMA': '1.14478', 'Leptin': '21.78',
     'Adiponectin': '17.86', 'Resistin': '4.82', 'MCP.1': '195.94', 'Classification': '2'},
    {'Age': '46', 'BMI': '22.21', 'Glucose': '86', 'Insulin': '36.94', 'HOMA': '7.836205333', 'Leptin': '10.16',
     'Adiponectin': '9.76', 'Resistin': '5.68', 'MCP.1': '312', 'Classification': '2'},
    {'Age': '43', 'BMI': '26.5625', 'Glucose': '101', 'Insulin': '10.555', 'HOMA': '2.629602333', 'Leptin': '9.8',
     'Adiponectin': '6.420295', 'Resistin': '16.1', 'MCP.1': '806.724', 'Classification': '2'},
    {'Age': '55', 'BMI': '31.97501487', 'Glucose': '92', 'Insulin': '16.635', 'HOMA': '3.775036', 'Leptin': '37.2234',
     'Adiponectin': '11.018455', 'Resistin': '7.16514', 'MCP.1': '483.377', 'Classification': '2'},
    {'Age': '43', 'BMI': '31.25', 'Glucose': '103', 'Insulin': '4.328', 'HOMA': '1.099600533', 'Leptin': '25.7816',
     'Adiponectin': '12.71896', 'Resistin': '38.6531', 'MCP.1': '775.322', 'Classification': '2'},
    {'Age': '86', 'BMI': '26.66666667', 'Glucose': '201', 'Insulin': '41.611', 'HOMA': '20.6307338', 'Leptin': '47.647',
     'Adiponectin': '5.357135', 'Resistin': '24.3701', 'MCP.1': '1698.44', 'Classification': '2'},
    {'Age': '41', 'BMI': '26.6727633', 'Glucose': '97', 'Insulin': '22.033', 'HOMA': '5.271762467', 'Leptin': '44.7059',
     'Adiponectin': '13.494865', 'Resistin': '27.8325', 'MCP.1': '783.796', 'Classification': '2'},
    {'Age': '59', 'BMI': '28.67262608', 'Glucose': '77', 'Insulin': '3.188', 'HOMA': '0.605507467', 'Leptin': '17.022',
     'Adiponectin': '16.44048', 'Resistin': '31.6904', 'MCP.1': '910.489', 'Classification': '2'},
    {'Age': '81', 'BMI': '31.64036818', 'Glucose': '100', 'Insulin': '9.669', 'HOMA': '2.38502', 'Leptin': '38.8066',
     'Adiponectin': '10.636525', 'Resistin': '29.5583', 'MCP.1': '426.175', 'Classification': '2'},
    {'Age': '48', 'BMI': '32.46191136', 'Glucose': '99', 'Insulin': '28.677', 'HOMA': '7.0029234', 'Leptin': '46.076',
     'Adiponectin': '21.57', 'Resistin': '10.15726', 'MCP.1': '738.034', 'Classification': '2'},
    {'Age': '71', 'BMI': '25.51020408', 'Glucose': '112', 'Insulin': '10.395', 'HOMA': '2.871792', 'Leptin': '19.0653',
     'Adiponectin': '5.4861', 'Resistin': '42.7447', 'MCP.1': '799.898', 'Classification': '2'},
    {'Age': '42', 'BMI': '29.296875', 'Glucose': '98', 'Insulin': '4.172', 'HOMA': '1.008511467', 'Leptin': '12.2617',
     'Adiponectin': '6.695585', 'Resistin': '53.6717', 'MCP.1': '1041.843', 'Classification': '2'},
    {'Age': '65', 'BMI': '29.666548', 'Glucose': '85', 'Insulin': '14.649', 'HOMA': '3.071407', 'Leptin': '26.5166',
     'Adiponectin': '7.28287', 'Resistin': '19.46324', 'MCP.1': '1698.44', 'Classification': '2'},
    {'Age': '48', 'BMI': '28.125', 'Glucose': '90', 'Insulin': '2.54', 'HOMA': '0.56388', 'Leptin': '15.5325',
     'Adiponectin': '10.22231', 'Resistin': '16.11032', 'MCP.1': '1698.44', 'Classification': '2'},
    {'Age': '85', 'BMI': '27.68877813', 'Glucose': '196', 'Insulin': '51.814', 'HOMA': '25.05034187',
     'Leptin': '70.8824', 'Adiponectin': '7.901685', 'Resistin': '55.2153', 'MCP.1': '1078.359', 'Classification': '2'},
    {'Age': '48', 'BMI': '31.25', 'Glucose': '199', 'Insulin': '12.162', 'HOMA': '5.9699204', 'Leptin': '18.1314',
     'Adiponectin': '4.104105', 'Resistin': '53.6308', 'MCP.1': '1698.44', 'Classification': '2'},
    {'Age': '58', 'BMI': '29.15451895', 'Glucose': '139', 'Insulin': '16.582', 'HOMA': '5.685415067',
     'Leptin': '22.8884', 'Adiponectin': '10.26266', 'Resistin': '13.97399', 'MCP.1': '923.886', 'Classification': '2'},
    {'Age': '40', 'BMI': '30.83653053', 'Glucose': '128', 'Insulin': '41.894', 'HOMA': '13.22733227',
     'Leptin': '31.0385', 'Adiponectin': '6.160995', 'Resistin': '17.55503', 'MCP.1': '638.261', 'Classification': '2'},
    {'Age': '82', 'BMI': '31.21748179', 'Glucose': '100', 'Insulin': '18.077', 'HOMA': '4.458993333',
     'Leptin': '31.6453', 'Adiponectin': '9.92365', 'Resistin': '19.94687', 'MCP.1': '994.316', 'Classification': '2'},
    {'Age': '52', 'BMI': '30.8012487', 'Glucose': '87', 'Insulin': '30.212', 'HOMA': '6.4834952', 'Leptin': '29.2739',
     'Adiponectin': '6.26854', 'Resistin': '24.24591', 'MCP.1': '764.667', 'Classification': '2'},
    {'Age': '49', 'BMI': '32.46191136', 'Glucose': '134', 'Insulin': '24.887', 'HOMA': '8.225983067',
     'Leptin': '42.3914', 'Adiponectin': '10.79394', 'Resistin': '5.768', 'MCP.1': '656.393', 'Classification': '2'},
    {'Age': '60', 'BMI': '31.23140988', 'Glucose': '131', 'Insulin': '30.13', 'HOMA': '9.736007333', 'Leptin': '37.843',
     'Adiponectin': '8.40443', 'Resistin': '11.50005', 'MCP.1': '396.021', 'Classification': '2'},
    {'Age': '49', 'BMI': '29.77777778', 'Glucose': '70', 'Insulin': '8.396', 'HOMA': '1.449709333', 'Leptin': '51.3387',
     'Adiponectin': '10.73174', 'Resistin': '20.76801', 'MCP.1': '602.486', 'Classification': '2'},
    {'Age': '44', 'BMI': '27.88761707', 'Glucose': '99', 'Insulin': '9.208', 'HOMA': '2.2485936', 'Leptin': '12.6757',
     'Adiponectin': '5.47817', 'Resistin': '23.03306', 'MCP.1': '407.206', 'Classification': '2'},
    {'Age': '40', 'BMI': '27.63605442', 'Glucose': '103', 'Insulin': '2.432', 'HOMA': '0.617890133',
     'Leptin': '14.3224', 'Adiponectin': '6.78387', 'Resistin': '26.0136', 'MCP.1': '293.123', 'Classification': '2'},
    {'Age': '71', 'BMI': '27.91551882', 'Glucose': '104', 'Insulin': '18.2', 'HOMA': '4.668906667', 'Leptin': '53.4997',
     'Adiponectin': '1.65602', 'Resistin': '49.24184', 'MCP.1': '256.001', 'Classification': '2'},
    {'Age': '69', 'BMI': '28.44444444', 'Glucose': '108', 'Insulin': '8.808', 'HOMA': '2.3464512', 'Leptin': '14.7485',
     'Adiponectin': '5.288025', 'Resistin': '16.48508', 'MCP.1': '353.568', 'Classification': '2'},
    {'Age': '74', 'BMI': '28.65013774', 'Glucose': '88', 'Insulin': '3.012', 'HOMA': '0.6538048', 'Leptin': '31.1233',
     'Adiponectin': '7.65222', 'Resistin': '18.35574', 'MCP.1': '572.401', 'Classification': '2'},
    {'Age': '66', 'BMI': '26.5625', 'Glucose': '89', 'Insulin': '6.524', 'HOMA': '1.432235467', 'Leptin': '14.9084',
     'Adiponectin': '8.42996', 'Resistin': '14.91922', 'MCP.1': '269.487', 'Classification': '2'},
    {'Age': '65', 'BMI': '30.91557669', 'Glucose': '97', 'Insulin': '10.491', 'HOMA': '2.5101466', 'Leptin': '44.0217',
     'Adiponectin': '3.71009', 'Resistin': '20.4685', 'MCP.1': '396.648', 'Classification': '2'},
    {'Age': '72', 'BMI': '29.13631634', 'Glucose': '83', 'Insulin': '10.949', 'HOMA': '2.241625267',
     'Leptin': '26.8081', 'Adiponectin': '2.78491', 'Resistin': '14.76966', 'MCP.1': '232.018', 'Classification': '2'},
    {'Age': '57', 'BMI': '34.83814777', 'Glucose': '95', 'Insulin': '12.548', 'HOMA': '2.940414667',
     'Leptin': '33.1612', 'Adiponectin': '2.36495', 'Resistin': '9.9542', 'MCP.1': '655.834', 'Classification': '2'},
    {'Age': '73', 'BMI': '37.109375', 'Glucose': '134', 'Insulin': '5.636', 'HOMA': '1.862885867', 'Leptin': '41.4064',
     'Adiponectin': '3.335665', 'Resistin': '6.89235', 'MCP.1': '788.902', 'Classification': '2'},
    {'Age': '45', 'BMI': '29.38475666', 'Glucose': '90', 'Insulin': '4.713', 'HOMA': '1.046286', 'Leptin': '23.8479',
     'Adiponectin': '6.644245', 'Resistin': '15.55625', 'MCP.1': '621.273', 'Classification': '2'},
    {'Age': '46', 'BMI': '33.18', 'Glucose': '92', 'Insulin': '5.75', 'HOMA': '1.304866667', 'Leptin': '18.69',
     'Adiponectin': '9.16', 'Resistin': '8.89', 'MCP.1': '209.19', 'Classification': '2'},
    {'Age': '68', 'BMI': '35.56', 'Glucose': '131', 'Insulin': '8.15', 'HOMA': '2.633536667', 'Leptin': '17.87',
     'Adiponectin': '11.9', 'Resistin': '4.19', 'MCP.1': '198.4', 'Classification': '2'},
    {'Age': '75', 'BMI': '30.48', 'Glucose': '152', 'Insulin': '7.01', 'HOMA': '2.628282667', 'Leptin': '50.53',
     'Adiponectin': '10.06', 'Resistin': '11.73', 'MCP.1': '99.45', 'Classification': '2'},
    {'Age': '54', 'BMI': '36.05', 'Glucose': '119', 'Insulin': '11.91', 'HOMA': '3.495982', 'Leptin': '89.27',
     'Adiponectin': '8.01', 'Resistin': '5.06', 'MCP.1': '218.28', 'Classification': '2'},
    {'Age': '45', 'BMI': '26.85', 'Glucose': '92', 'Insulin': '3.33', 'HOMA': '0.755688', 'Leptin': '54.68',
     'Adiponectin': '12.1', 'Resistin': '10.96', 'MCP.1': '268.23', 'Classification': '2'},
    {'Age': '62', 'BMI': '26.84', 'Glucose': '100', 'Insulin': '4.53', 'HOMA': '1.1174', 'Leptin': '12.45',
     'Adiponectin': '21.42', 'Resistin': '7.32', 'MCP.1': '330.16', 'Classification': '2'},
    {'Age': '65', 'BMI': '32.05', 'Glucose': '97', 'Insulin': '5.73', 'HOMA': '1.370998', 'Leptin': '61.48',
     'Adiponectin': '22.54', 'Resistin': '10.33', 'MCP.1': '314.05', 'Classification': '2'},
    {'Age': '72', 'BMI': '25.59', 'Glucose': '82', 'Insulin': '2.82', 'HOMA': '0.570392', 'Leptin': '24.96',
     'Adiponectin': '33.75', 'Resistin': '3.27', 'MCP.1': '392.46', 'Classification': '2'},
    {'Age': '86', 'BMI': '27.18', 'Glucose': '138', 'Insulin': '19.91', 'HOMA': '6.777364', 'Leptin': '90.28',
     'Adiponectin': '14.11', 'Resistin': '4.35', 'MCP.1': '90.09', 'Classification': '2'}]
assert len(data) == 116
label = "Classification"

data_unlabeled = [[float(v) for k, v in i.items() if k != label] for i in data]
x = scale(data_unlabeled)
x_normalised = normalize(data_unlabeled, axis=0)
x_unscaled = np.array(data_unlabeled)
x_all_vars = scale([[float(v) for k, v in i.items() if k != label] for i in data_all_vars])
y = np.array([1 if int(i[label]) == 2 else 0 for i in data])  # Converting labels to 0 and 1 (for Keras).
dimensions = len(x[0])
