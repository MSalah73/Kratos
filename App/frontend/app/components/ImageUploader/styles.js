import EStyleSheet from 'react-native-extended-stylesheet';

export default EStyleSheet.create({
	container: {
		alignItems: 'center',
		justifyContent: 'center',
	},
	containerIcons: {
		alignItems: 'center',
		justifyContent: 'center',
		flexDirection: 'row',
		paddingLeft: 9,
	},
	containerIcon: {
		padding: 8,
	},
	containerImage: {
		alignItems: 'center',
		justifyContent: 'center',
		width: 200,
		height: 200,
		padding: 8,
	},
	containerText: {
		fontWeight: '600',
		fontSize: 18,
		letterSpacing: 0.5,
		marginTop: 0,
		textAlign: 'center',
		//color: '$white',
	},
});
