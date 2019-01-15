import React from 'react';
import ProbTypes from 'prop-types';
import { View } from 'react-native'; 
import styles from './styles'

const Container = ({children}) => (
	<View style={styles.container}>
		{children}
	</View>
);

Container.probTypes = {
	Children: ProbTypes.any,
};

export default Container;