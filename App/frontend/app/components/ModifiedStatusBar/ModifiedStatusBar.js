import React from 'react';
import { View, StatusBar } from 'react-native';
import styles from './styles'

const ModifiedStatusBar = ({...props}) => (
	<View style={styles.container}>
		<StatusBar {...props} />
	</View>
);

export default ModifiedStatusBar;
