import EStyleSheet from 'react-native-extended-stylesheet';
import {Dimensions} from 'react-native';

const imageWidth = Dimensions.get('window').width / 2
const imageHeight = Dimensions.get('window').height / 2

export default EStyleSheet.create({
	container: {
		alignItems: 'center',
	},
	containerImage: {
		alignItems: 'center',
		justifyContent: 'center',
		width: imageWidth/ 3,
		height: imageHeight /3,
	},
	containerText: {
		fontWeight: '600',
		fontSize: 28,
		letterSpacing: 0.5,
		marginTop: -15,
		//color: '$white',
	},
});
